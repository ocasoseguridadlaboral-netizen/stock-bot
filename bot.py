import os, json, re, logging, sys, unicodedata
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
import gspread
import pkg_resources

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ==== IA (OpenAI) opcional ====
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ----------------- CONFIG -----------------
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "").strip()
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

SHEET_PRODUCTS_NAME = os.getenv("SHEET_PRODUCTS_NAME", "Productos")
SHEET_MOVS_NAME = os.getenv("SHEET_MOVS_NAME", "Movimientos")

# Cabecera base; 'Categoria' es OPCIONAL (si existe, se usa)
PRODUCTS_HEADERS = ["Codigo", "Categoria", "Producto", "Talle", "Color", "Costo", "Precio", "Stock", "Minimo", "SKU"]
MOVS_HEADERS = ["Fecha", "Tipo", "Codigo", "Producto", "Talle", "Color", "Descripcion", "Cantidad", "PrecioVenta", "CostoUnitario", "Ganancia"]

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger("stock-bot")

# ----------------- NORMALIZACIÓN / UTIL -----------------
def _normalize(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s)

_SINGULAR_MAP = {
    "pantalones":"pantalon", "botines":"botin", "zapatillas":"zapatilla",
    "medias":"media", "camisas":"camisa", "remeras":"remera", "jeans":"jean",
    "gorras":"gorra", "camperas":"campera", "buzos":"buzo", "bermudas":"bermuda",
    "shorts":"bermuda", "short":"bermuda"
}
def _singularize(word: str) -> str:
    w = _normalize(word)
    if w in _SINGULAR_MAP: return _SINGULAR_MAP[w]
    if w.endswith("es") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w

def _money(x: float) -> str:
    return f"${x:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")

def _is_probable_code(text: str) -> bool:
    t = _normalize(text)
    return bool(re.fullmatch(r"[a-z0-9\-]{1,20}", t))

# ----------------- GOOGLE SHEETS -----------------
def _gs_client():
    svc_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if not svc_json:
        raise RuntimeError("Falta GOOGLE_SERVICE_ACCOUNT_JSON")
    info = json.loads(svc_json)
    return gspread.service_account_from_dict(info)

def _open_sheet():
    gc = _gs_client()
    sh = gc.open_by_key(GOOGLE_SHEET_ID)
    try:
        ws_prod = sh.worksheet(SHEET_PRODUCTS_NAME)
    except gspread.WorksheetNotFound:
        ws_prod = sh.add_worksheet(SHEET_PRODUCTS_NAME, rows=1000, cols=len(PRODUCTS_HEADERS))
        ws_prod.append_row(PRODUCTS_HEADERS)
    try:
        ws_movs = sh.worksheet(SHEET_MOVS_NAME)
    except gspread.WorksheetNotFound:
        ws_movs = sh.add_worksheet(SHEET_MOVS_NAME, rows=1000, cols=len(MOVS_HEADERS))
        ws_movs.append_row(MOVS_HEADERS)
    return sh, ws_prod, ws_movs

def _read_products(ws):
    values = ws.get_all_values()
    if not values:
        ws.append_row(PRODUCTS_HEADERS)
        return [], {}
    headers = [h.strip() for h in values[0]]
    idx = {h.lower(): i for i, h in enumerate(headers)}
    rows = values[1:]
    products = []
    for r_i, row in enumerate(rows, start=2):
        if not any(row):
            continue

        def get_text(col):
            key = col.lower()
            return str(row[idx[key]]).strip() if key in idx and idx[key] < len(row) else ""

        def get_num(col):
            key = col.lower()
            try: return float(str(row[idx[key]]).replace(",", ".") or 0)
            except: return 0

        p = {
            "row": r_i,
            "Codigo": get_text("Codigo"),
            "Categoria": get_text("Categoria") if "categoria" in idx else "",  # opcional
            "Producto": get_text("Producto"),
            "Talle": get_text("Talle"),
            "Color": get_text("Color"),
            "Costo": get_num("Costo"),
            "Precio": get_num("Precio"),
            "Stock": int(get_num("Stock")),
            "Minimo": int(get_num("Minimo")),
            "SKU": get_text("SKU")
        }
        # Compat: si no hay Categoria pero existe Descripcion, intentar inferir
        if not p["Categoria"] and "descripcion" in idx:
            desc = _normalize(str(row[idx["descripcion"]]))
            for w in ["pantalon","bermuda","botin","camisa","remera","campera","chaleco","buzo","zapatilla","guante","jean","gorra"]:
                if re.search(rf"\b{w}\b", desc):
                    p["Categoria"] = w
                    break

        products.append(p)
    return products, idx

# ----------------- BÚSQUEDAS -----------------
def _find_by_code(code: str, products: List[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
    c = _normalize(code)
    for p in products:
        if _normalize(p.get("Codigo","")) == c or _normalize(p.get("SKU","")) == c:
            return p
    return None

def _catalogo_valores(products: List[Dict[str,Any]]) -> Tuple[List[str], List[str], List[str], List[str]]:
    categorias = sorted({ _normalize(p.get("Categoria","")) for p in products if p.get("Categoria") }, key=len, reverse=True)
    prods = sorted({ _normalize(p.get("Producto","")) for p in products if p.get("Producto") }, key=len, reverse=True)
    colores = sorted({ _normalize(p.get("Color","")) for p in products if p.get("Color") }, key=len, reverse=True)
    talles = sorted({ _normalize(p.get("Talle","")) for p in products if p.get("Talle") }, key=len, reverse=True)
    return categorias, prods, colores, talles

def _pick_one_from(text_nrm: str, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c and c in text_nrm:
            return c
    return None

def _detect_categoria_from_text(text_nrm: str) -> Optional[str]:
    for w in ["pantalon","bermuda","botin","camisa","remera","campera","chaleco","buzo","zapatilla","guante","jean","gorra"]:
        if re.search(rf"\b{w}s?\b", text_nrm):
            return w
    return None

def _extract_entities_from_query(products: List[Dict[str,Any]], text: str) -> Dict[str, Optional[str]]:
    t = _normalize(text)
    tokens = [_singularize(tok) for tok in t.split()]
    t_sing = " ".join(tokens)

    cats_cat, prods_cat, colores_cat, talles_cat = _catalogo_valores(products)

    categoria = _pick_one_from(" " + t_sing + " ", cats_cat) or _detect_categoria_from_text(t_sing)
    producto = _pick_one_from(t_sing, prods_cat)
    color = _pick_one_from(t_sing, colores_cat)

    # talle: por palabra clave o match directo
    m = re.search(r"\b(?:talle|t)\s*([a-z0-9\-]+)\b", t_sing)
    if m:
        talle = m.group(1)
    else:
        talle = None
        for tok in tokens:
            if tok in talles_cat:
                talle = tok; break
            if re.fullmatch(r"\d{2}", tok) and tok in talles_cat:
                talle = tok; break

    # si el usuario dijo "codigo 79"
    m2 = re.search(r"\bcod(?:igo)?\s*([a-z0-9\-]+)\b", t_sing)
    codigo = m2.group(1) if m2 else None

    return {"Categoria": categoria, "Producto": producto, "Talle": talle, "Color": color, "Codigo": codigo}

def _filtrar_por_campos(products: List[Dict[str,Any]], f: Dict[str, Optional[str]]) -> List[Dict[str,Any]]:
    """
    NUEVO: ignora un campo si NO viene en el filtro, o si la CELDA de la fila está vacía.
    Así no falla cuando la hoja no tiene 'Categoria' o está en blanco.
    """
    def ok(p, k):
        q = _normalize(f.get(k) or "")
        if not q:
            return True  # no pidieron filtrar por ese campo
        v = _normalize(p.get(k, "") or "")
        if not v:
            return True  # la fila no tiene ese dato → no descarto por esto
        return q == v

    # Si viene código, priorizamos match por código/SKU
    if f.get("Codigo"):
        p = _find_by_code(f["Codigo"], products)
        return [p] if p else []

    return [
        p for p in products
        if ok(p, "Categoria") and ok(p, "Producto") and ok(p, "Talle") and ok(p, "Color")
    ]

def _buscar_textual(products: List[Dict[str,Any]], query: str) -> List[Dict[str,Any]]:
    q = _normalize(query)
    toks = set(q.split())
    out = []
    for p in products:
        texto = _normalize(" ".join([
            p.get("Codigo",""), p.get("Categoria",""), p.get("Producto",""),
            p.get("Talle",""), p.get("Color",""), p.get("SKU","")
        ]))
        if q in texto or toks.issubset(set(texto.split())):
            out.append(p)
    return out

# ----------------- INTENCIÓN / ENTIDADES (IA opcional) -----------------
VERBOS_SUMA = {"compre","compré","compra","agrega","agregá","agregar","sumar","sumá","suma","entrada","entraron","ingresa","ingresá","agrego","ingreso"}
VERBOS_RESTA = {"vendi","vendí","venta","vendimos","desconta","descontá","descontar","restar","resta","salida","salieron","retiro","retiré","descuento"}
UNDO_WORDS = {"anular","deshacer","undo","revertir","volver atras","cancelar ultima","cancelá ultima","anula","deshace"}

def _detectar_intencion(text_nrm: str) -> Optional[str]:
    if any(w in text_nrm for w in UNDO_WORDS):
        return "anular_ultimo"
    tokens = text_nrm.split()
    if any(v in tokens for v in VERBOS_RESTA): return "descontar"
    if any(v in tokens for v in VERBOS_SUMA): return "agregar"
    if re.search(r"\bvend", text_nrm): return "descontar"
    if re.search(r"\bcompr", text_nrm): return "agregar"
    return None

def _extraer_cantidad(text_nrm: str) -> int:
    # Evitar confundir TALLE con cantidad: ignoramos números que siguen a 'talle'/'t'
    tokens = text_nrm.split()
    candidatos = []
    for i, tok in enumerate(tokens):
        if re.fullmatch(r"\d{1,4}", tok):
            # si viene justo después de 'talle' o 't', NO es cantidad
            if i > 0 and tokens[i-1] in {"talle","t"}:
                continue
            candidatos.append(int(tok))
    if candidatos:
        return candidatos[0]
    return 1

def _extraer_precio(text_nrm: str) -> Optional[float]:
    m = re.search(r"\$?\s*([\d\.]{1,3}(?:[\.\s]?\d{3})*(?:[\,\.]\d{1,2})?)", text_nrm)
    if m:
        val = m.group(1).replace(".", "").replace(" ", "").replace(",", ".")
        try:
            f = float(val)
            return f if f > 0 else None
        except:
            return None
    return None

def _limpiar_codeblock(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.startswith("json"):
            s = s[4:].strip()
    return s

def _nlp_parse(text: str) -> Dict[str, Any]:
    t = _normalize(text)
    accion = _detectar_intencion(t)
    if accion == "anular_ultimo":
        return {"intent":"anular_ultimo", "data":{}}

    cantidad = _extraer_cantidad(t)
    precio = _extraer_precio(t)
    base = {"intent":"ajustar_stock","data":{"accion":accion,"cantidad":cantidad,"query":text,"precio_venta":precio}}

    # IA opcional: estructura JSON limpia
    if OPENAI_API_KEY and OpenAI:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            system = (
                "Sos un parser de español para control de stock. "
                "Devolvé JSON con 'intent' y 'data'. intents: 'ajustar_stock','reporte','faltantes','ganancias','anular_ultimo'. "
                "Si el texto implica anular/deshacer, devolvé intent='anular_ultimo'. "
                "Para 'ajustar_stock' incluí: {'accion':'agregar'|'descontar','cantidad':int,'query':str,'precio_venta':float|null, "
                "'categoria':str,'producto':str,'talle':str,'color':str,'codigo':str}"
            )
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":system},{"role":"user","content":text}],
                response_format={"type":"json_object"},
                temperature=0
            )
            content = _limpiar_codeblock(resp.choices[0].message.content or "")
            data = json.loads(content)
            d = data.get("data",{})
            if not data.get("intent"): data["intent"] = "ajustar_stock"
            # completar faltantes con heurísticas locales
            if data["intent"] == "ajustar_stock":
                if not d.get("accion"): d["accion"] = accion
                if not d.get("cantidad"): d["cantidad"] = cantidad
                if "precio_venta" not in d: d["precio_venta"] = precio
                data["data"] = d
            return data
        except Exception as e:
            log.warning(f"OpenAI deshabilitado por error: {e}")

    return base

# ============= MULTI-ITEM PARSER =============
def _split_into_items(text: str) -> List[str]:
    """Divide un mensaje en items por ',' o ' y ' sin romper números."""
    t = " " + _normalize(text) + " "
    # reemplazo ' y ' por coma para unificar separadores
    t = re.sub(r"\s+y\s+", ",", t)
    # partir por comas
    partes = [p.strip() for p in t.split(",")]
    return [p for p in partes if p]

def _build_item_from_text(products, text: str, accion_global: Optional[str]) -> Dict[str, Any]:
    """Crea un dict de ajuste a partir de un item (puede heredar la acción global)."""
    parsed = _nlp_parse(text)
    data = parsed.get("data", {})
    accion = data.get("accion") or accion_global or "descontar"
    cantidad = int(data.get("cantidad") or _extraer_cantidad(_normalize(text)) or 1)
    precio_venta = data.get("precio_venta")

    # entidades
    filtro = _extract_entities_from_query(products, text)
    # si IA trajo campos, priorizalos
    for k_src, k_dst in [("categoria","Categoria"), ("producto","Producto"), ("talle","Talle"), ("color","Color"), ("codigo","Codigo")]:
        v = data.get(k_src)
        if v and not filtro.get(k_dst):
            filtro[k_dst] = _normalize(v)
    return {"accion":accion, "cantidad":cantidad, "precio_venta":precio_venta, "filtro":filtro, "texto":text}

# ----------------- MOVIMIENTOS / STOCK -----------------
def _update_stock(ws, prod, new_stock, idx):
    col = idx["stock"] + 1 if "stock" in idx else 8  # H por defecto
    ws.update_cell(prod["row"], col, new_stock)

def _append_movement(ws_movs, tipo, prod, cantidad, precio_venta, costo_unit):
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ganancia = (precio_venta - costo_unit) * abs(int(cantidad)) if tipo == "salida" else 0
    ws_movs.append_row([
        fecha, tipo, prod.get("Codigo",""), prod.get("Producto",""),
        prod.get("Talle",""), prod.get("Color",""),
        prod.get("Categoria","") or "",  # usamos 'Descripcion' para categoria
        int(cantidad), float(precio_venta or 0), float(costo_unit or 0), float(ganancia)
    ])

def _sum_ganancias(ws_movs) -> float:
    values = ws_movs.get_all_values()
    if not values or len(values) < 2: return 0
    headers = [h.strip().lower() for h in values[0]]
    if "ganancia" not in headers: return 0
    idx_g = headers.index("ganancia")
    total = 0
    for r in values[1:]:
        try: total += float(str(r[idx_g]).replace(",", ".") or 0)
        except: pass
    return total

def _descripcion_corta(p: Dict[str,Any]) -> str:
    cat = p.get("Categoria","") or ""
    return f"{cat} {p.get('Producto','')} {p.get('Talle','')} {p.get('Color','')}".strip()

# ----------------- SLOT-FILLING -----------------
def _estado_pendiente(context: ContextTypes.DEFAULT_TYPE) -> Optional[Dict[str,Any]]:
    return context.user_data.get("pending_adjust")

def _set_pendiente(context: ContextTypes.DEFAULT_TYPE, data: Optional[Dict[str,Any]]):
    if data is None:
        context.user_data.pop("pending_adjust", None)
    else:
        context.user_data["pending_adjust"] = data

def _siguiente_slot_faltante(filtro: Dict[str, Optional[str]]) -> Optional[str]:
    for k in ["Categoria","Producto","Talle","Color"]:
        if not filtro.get(k):
            return k
    return None

async def _preguntar_slot(update: Update, slot: str, products: List[Dict[str,Any]]):
    if slot == "Categoria":
        await update.message.reply_text("¿Es *pantalon*, *bermuda*, *botin*, etc.?", parse_mode="Markdown")
    elif slot == "Producto":
        await update.message.reply_text("¿Qué *producto* es? (ej. 'clasico', 'cargo')", parse_mode="Markdown")
    elif slot == "Talle":
        await update.message.reply_text("¿Qué *talle*?", parse_mode="Markdown")
    elif slot == "Color":
        await update.message.reply_text("¿Qué *color*?", parse_mode="Markdown")
    else:
        await update.message.reply_text("Necesito un dato más…")

async def _resolver_y_ajustar(update, context, accion, cantidad, precio_venta, filtro, products, idx, ws_prod, ws_movs):
    # 1) Intento por filtro exacto (tolerante a vacíos)
    candidatos = _filtrar_por_campos(products, filtro)
    if not candidatos:
        # 2) Intento textual con campos PRESENTES (sin obligar categoria si no está)
        q_parts = []
        for k in ("Producto","Talle","Color","Codigo","Categoria"):
            if filtro.get(k):
                q_parts.append(str(filtro[k]))
        candidatos = _buscar_textual(products, " ".join(q_parts))

    if not candidatos:
        slot = _siguiente_slot_faltante(filtro)
        if slot:
            await _preguntar_slot(update, slot, products)
            _set_pendiente(context, {"accion":accion,"cantidad":cantidad,"precio_venta":precio_venta,"filtro":filtro,"t":datetime.now().timestamp()})
            return
        await update.message.reply_text("No encontré coincidencias. Indicá *categoria, producto, talle y color*.", parse_mode="Markdown")
        return

    if len(candidatos) > 1:
        lista = "\n".join([f"- {_descripcion_corta(p)} código: {p.get('Codigo') or 's/cod'}" for p in candidatos[:10]])
        await update.message.reply_text(
            f"Encontré varias coincidencias:\n{lista}\n\nDecime el *código* exacto o especificá mejor (p. ej. 'pantalon clasico 42 verde').",
            parse_mode="Markdown"
        )
        _set_pendiente(context, {"accion":accion,"cantidad":cantidad,"precio_venta":precio_venta,"filtro":filtro,"t":datetime.now().timestamp()})
        return

    # === ÚNICO PRODUCTO → AJUSTAR ===
    prod = candidatos[0]
    delta = abs(cantidad) if accion == "agregar" else -abs(cantidad)
    new_stock = max(0, prod["Stock"] + delta)
    _update_stock(ws_prod, prod, new_stock, idx)
    tipo = "entrada" if delta > 0 else "salida"
    precio_v = float(precio_venta or prod["Precio"] or 0)
    costo = float(prod["Costo"] or 0)
    _append_movement(ws_movs, tipo, prod, abs(delta), precio_v, costo)

    _set_pendiente(context, None)  # limpiar pendiente

    msg = f"✅ {('Sumé' if delta>0 else 'Desconté')} {abs(delta)} de {_descripcion_corta(prod)} (stock {prod['Stock']}→{new_stock})"
    if tipo == "salida" and precio_v and costo:
        msg += f"\nGanancia estimada: {_money((precio_v - costo)*abs(delta))}"
    await update.message.reply_text(msg)

# ----------------- UNDO ÚLTIMO MOVIMIENTO -----------------
def _get_headers_index(values: List[List[str]]) -> Dict[str,int]:
    headers = [h.strip().lower() for h in (values[0] if values else [])]
    return {h:i for i,h in enumerate(headers)}

def _find_product_by_rowdata(products, codigo, producto, talle, color, categoria) -> Optional[Dict[str,Any]]:
    # 1) por código
    if codigo:
        p = _find_by_code(codigo, products)
        if p: return p
    # 2) por coincidencia campos
    f = {"Categoria":categoria or "", "Producto":producto or "", "Talle":talle or "", "Color":color or ""}
    cands = _filtrar_por_campos(products, f)
    if cands: return cands[0]
    # 3) textual
    q = " ".join([x for x in [categoria, producto, talle, color, codigo] if x])
    cands = _buscar_textual(products, q)
    return cands[0] if cands else None

async def _cmd_anular(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        sh, ws_prod, ws_movs = _open_sheet()
        values = ws_movs.get_all_values()
        if not values or len(values) < 2:
            await update.message.reply_text("No hay movimientos para anular.")
            return
        idx = _get_headers_index(values)
        last_row_num = len(values)  # 1-based
        last = values[-1]
        def get(col):
            if col in idx and idx[col] < len(last):
                return last[idx[col]]
            return ""

        tipo = _normalize(get("tipo"))        # entrada | salida
        codigo = get("codigo").strip()
        producto = get("producto").strip()
        talle = get("talle").strip()
        color = get("color").strip()
        categoria = get("descripcion").strip()  # acá guardamos categoría

        products, pidx = _read_products(ws_prod)
        prod = _find_product_by_rowdata(products, codigo, producto, talle, color, categoria)
        if not prod:
            await update.message.reply_text("No pude ubicar el producto del último movimiento para revertir.")
            return

        try:
            cant = int(float(get("cantidad").replace(",", ".") or "0"))
        except:
            cant = 0
        if cant <= 0:
            await update.message.reply_text("El último movimiento no tiene cantidad válida para revertir.")
            return

        # Revertir: si fue entrada, ahora salida; si fue salida, ahora entrada
        delta = -abs(cant) if tipo == "entrada" else +abs(cant)

        new_stock = max(0, prod["Stock"] + delta)
        _update_stock(ws_prod, prod, new_stock, pidx)
        # Borrar la fila de movimientos
        ws_movs.delete_rows(last_row_num)

        signo = "Desconté" if delta < 0 else "Sumé"
        await update.message.reply_text(f"↩️ Anulado. {signo} {abs(delta)} de {_descripcion_corta(prod)} (stock {prod['Stock']}→{new_stock}).")
    except Exception as e:
        log.error(f"Anular error: {e}")
        await update.message.reply_text(f"❌ No se pudo anular: {e}")

# ----------------- HANDLERS BÁSICOS -----------------
async def _cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "👋 *Bot de Control de Stock*\n\n"
        "Ejemplos:\n"
        "• 'Compré 4 pantalones cargo 50 verde'\n"
        "• 'Vendí 2 bermudas cargo 50 verde a $34000'\n"
        "• 'Código 79 vendí 1'\n"
        "• *Múltiples*: 'vendí 2 pantalones clásico 44, una camisa 42 y una camisa 44'\n"
        "• 'Mostrame los faltantes' | 'Ganancias'\n"
        "• 'anular' o /anular → deshace el último movimiento\n\n"
        "Comandos: /reporte, /faltantes, /ganancias, /agregar, /anular, /ping, /version, /estado"
    )
    await update.message.reply_markdown(msg)

async def _cmd_reporte(update, context):
    _, ws, _ = _open_sheet()
    products, _ = _read_products(ws)
    if not products:
        await update.message.reply_text("No hay productos cargados.")
        return
    text = "📊 *Reporte de stock:*\n"
    for p in products[:300]:
        text += f"- {_descripcion_corta(p)} | Stock: {p['Stock']} (mín: {p['Minimo']}) | Precio: {_money(p['Precio'])}\n"
    await update.message.reply_markdown(text[:4000])

async def _cmd_faltantes(update, context):
    _, ws, _ = _open_sheet()
    products, _ = _read_products(ws)
    falt = [p for p in products if p["Stock"] <= p["Minimo"]]
    if not falt:
        await update.message.reply_text("✅ No hay faltantes.")
        return
    text = "🚨 *Bajo stock:*\n"
    for p in falt[:300]:
        text += f"- {_descripcion_corta(p)} ({p['Stock']} / mín {p['Minimo']})\n"
    await update.message.reply_markdown(text[:4000])

async def _cmd_ganancias(update, context):
    _, _, ws_movs = _open_sheet()
    total = _sum_ganancias(ws_movs)
    await update.message.reply_text(f"💰 Ganancia total: {_money(total)}")

# /agregar: alta de fila completa con tu orden de columnas (Categoria opcional)
async def _cmd_agregar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # /agregar codigo,categoria,producto,talle,color,costo,precio,stock,minimo,sku
        texto = " ".join(context.args)
        partes = [p.strip() for p in texto.split(",")]
        if len(partes) not in (9,10):
            raise ValueError("Cantidad de campos inválida")

        _, ws_prod, _ = _open_sheet()
        values = ws_prod.get_all_values()
        headers = [h.strip().lower() for h in values[0]] if values else []
        has_categoria = "categoria" in headers

        if len(partes) == 9:
            # sin categoria
            if has_categoria:
                # insertar vacío en Categoria
                partes = [partes[0], ""] + partes[1:]
        else:
            # 10 campos: ya trae categoria
            if not has_categoria:
                # descartar categoria si no existe columna
                partes = [partes[0]] + partes[2:]

        def to_float(x):
            try: return float(str(x).replace(",", "."))
            except: return 0.0
        def to_int(x):
            try: return int(float(str(x).replace(",", ".")))
            except: return 0

        if has_categoria and len(partes) != 10: raise ValueError("Se esperaban 10 campos (incluida Categoria).")
        if not has_categoria and len(partes) != 9: raise ValueError("Se esperaban 9 campos (sin Categoria).")

        # Costo/Precio/Stock/Minimo a tipos numéricos
        if has_categoria:
            partes[5] = to_float(partes[5])
            partes[6] = to_float(partes[6])
            partes[7] = to_int(partes[7])
            partes[8] = to_int(partes[8])
        else:
            partes[5] = to_float(partes[5])
            partes[6] = to_float(partes[6])
            partes[7] = to_int(partes[7])
            partes[8] = to_int(partes[8])

        ws_prod.append_row(partes, value_input_option="USER_ENTERED")
        await update.message.reply_text("✅ Producto agregado correctamente.")
    except Exception as e:
        log.error(f"/agregar error: {e}")
        await update.message.reply_text(
            "❌ Formato:\n/agregar codigo,categoria,producto,talle,color,costo,precio,stock,minimo,sku\n"
            "Si tu hoja no tiene *Categoria*, omitila (9 campos)."
        )

# ----------------- MENSAJES -----------------
async def _on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text or ""
    text_nrm = _normalize(text)

    # Atajo: si pide anular/deshacer en texto libre
    if any(w in text_nrm for w in UNDO_WORDS):
        await _cmd_anular(update, context)
        return

    sh, ws_prod, ws_movs = _open_sheet()
    products, idx = _read_products(ws_prod)

    # Detectar intención global (agregar/descontar) del texto completo
    intent_global = _detectar_intencion(text_nrm)
    if intent_global == "anular_ultimo":
        await _cmd_anular(update, context); return

    # Separar items por coma/“ y ”
    items = _split_into_items(text)
    if len(items) <= 1:
        # Caso simple → flujo previo con slot-filling
        parsed = _nlp_parse(text)
        intent = parsed.get("intent")
        data = parsed.get("data", {})

        if intent in ("reporte","faltantes","ganancias"):
            if intent == "reporte":
                await _cmd_reporte(update, context)
            elif intent == "faltantes":
                await _cmd_faltantes(update, context)
            else:
                await _cmd_ganancias(update, context)
            return

        accion = data.get("accion") or ("agregar" if re.search(r"\bcompr|agreg|sum", text_nrm) else "descontar")
        cantidad = int(data.get("cantidad") or _extraer_cantidad(text_nrm) or 1)
        precio_venta = data.get("precio_venta")

        filtro = _extract_entities_from_query(products, text)
        for k_src, k_dst in [("categoria","Categoria"), ("producto","Producto"), ("talle","Talle"), ("color","Color"), ("codigo","Codigo")]:
            v = data.get(k_src)
            if v and not filtro.get(k_dst):
                filtro[k_dst] = _normalize(v)

        await _resolver_y_ajustar(update, context, accion, cantidad, precio_venta, filtro, products, idx, ws_prod, ws_movs)
        return

    # Multi-ítem: procesar cada parte
    ajustes: List[Dict[str,Any]] = []
    for it in items:
        adj = _build_item_from_text(products, it, intent_global if intent_global in ("agregar","descontar") else None)
        ajustes.append(adj)

    # Ejecutar cada ajuste (con slot-filling si falta info)
    await update.message.reply_text(f"🧾 Detecté {len(ajustes)} ítems. Voy aplicando…")
    for adj in ajustes:
        await _resolver_y_ajustar(
            update, context,
            adj["accion"], adj["cantidad"], adj["precio_venta"],
            adj["filtro"], products, idx, ws_prod, ws_movs
        )

# ----------------- /ping /version /estado -----------------
async def _cmd_ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        sh, ws_prod, ws_movs = _open_sheet()
        _ = ws_prod.title; _ = ws_movs.title
        await update.message.reply_text(f"✅ Bot online\n🧾 Google Sheets OK → '{sh.title}'")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Bot activo, pero sin acceso a Sheets:\n{e}")

async def _cmd_version(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        pyver = sys.version.split()[0]
        libs = {
            "python-telegram-bot": pkg_resources.get_distribution("python-telegram-bot").version,
            "gspread": pkg_resources.get_distribution("gspread").version,
            "google-auth": pkg_resources.get_distribution("google-auth").version,
            "openai": pkg_resources.get_distribution("openai").version if OPENAI_API_KEY else "(deshabilitado)",
            "httpx": pkg_resources.get_distribution("httpx").version,
        }
        msg = f"🧠 *Versión del Bot*\nPython: `{pyver}`\n" + "\n".join([f"{k}: `{v}`" for k,v in libs.items()])
        await update.message.reply_markdown(msg)
    except Exception as e:
        await update.message.reply_text(f"⚠️ Error al obtener versiones: {e}")

async def _cmd_estado(update: Update, context: ContextTypes.DEFAULT_TYPE):
    lines = []
    try:
        sh, ws_prod, ws_movs = _open_sheet()
        products, _ = _read_products(ws_prod)
        falt = [p for p in products if p["Stock"] <= p["Minimo"]]
        lines.append(f"🧾 Sheets: OK → '{sh.title}'")
        lines.append(f"📦 Productos: {len(products)} | 🚨 Faltantes: {len(falt)}")
        mov_values = ws_movs.get_all_values()
        ult = []
        if mov_values and len(mov_values) > 1:
            for r in mov_values[-3:]:
                if r and len(r) >= 11:
                    ult.append(f"{r[0]} · {r[1]} · {r[3]} {r[4]} {r[5]} · cant {r[7]}")
        lines.append("📝 Últimos movimientos:" + ("\n  • " + "\n  • ".join(ult) if ult else " (sin registros)"))
    except Exception as e:
        lines.append(f"🧾 Sheets: ERROR → {e}")
    try:
        if OPENAI_API_KEY and OpenAI:
            client = OpenAI(api_key=OPENAI_API_KEY)
            _ = client.models.list()
            lines.append("🤖 IA (OpenAI): OK")
        else:
            lines.append("🤖 IA (OpenAI): deshabilitada (sin API key)")
    except Exception as e:
        lines.append(f"🤖 IA (OpenAI): ERROR → {e}")
    await update.message.reply_text("\n".join(lines)[:4000])

# ----------------- MAIN -----------------
def main():
    if not TELEGRAM_TOKEN or not GOOGLE_SHEET_ID:
        raise RuntimeError("Faltan variables de entorno: TELEGRAM_TOKEN y/o GOOGLE_SHEET_ID.")
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", _cmd_start))
    app.add_handler(CommandHandler("reporte", _cmd_reporte))
    app.add_handler(CommandHandler("faltantes", _cmd_faltantes))
    app.add_handler(CommandHandler("ganancias", _cmd_ganancias))
    app.add_handler(CommandHandler("agregar", _cmd_agregar))
    app.add_handler(CommandHandler("anular", _cmd_anular))
    app.add_handler(CommandHandler("ping", _cmd_ping))
    app.add_handler(CommandHandler("version", _cmd_version))
    app.add_handler(CommandHandler("estado", _cmd_estado))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_text))

    log.info("Bot corriendo…")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
