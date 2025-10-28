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
    "shorts":"bermuda", "short":"bermuda", "pantaloness":"pantalon"
}
def _singularize(word: str) -> str:
    w = _normalize(word)
    if w in _SINGULAR_MAP: return _SINGULAR_MAP[w]
    if w.endswith("es") and len(w) > 3: return w[:-2]
    if w.endswith("s") and len(w) > 3: return w[:-1]
    return w

def _money(x: float) -> str:
    return f"${x:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")

def _is_probable_code(text: str) -> bool:
    t = _normalize(text)
    return bool(re.fullmatch(r"[a-z0-9\-]{1,20}", t))

COLOR_ALIASES = {
    "caqui":"kaki", "kakhi":"kaki", "khaki":"kaki",
    "azulino":"azul", "azul marino":"azul", "marino":"azul",
    "blancco":"blanco", "negrro":"negro", "gris oscuro":"gris", "gris claro":"gris",
}
def _alias_color(c: str) -> str:
    c2 = _normalize(c)
    return COLOR_ALIASES.get(c2, c2)

NUMBER_WORDS = {
    "un":1,"uno":1,"una":1,"dos":2,"tres":3,"cuatro":4,"cinco":5,"seis":6,"siete":7,"ocho":8,"nueve":9,"diez":10
}

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
            "Categoria": get_text("Categoria") if "categoria" in idx else "",
            "Producto": get_text("Producto"),
            "Talle": get_text("Talle"),
            "Color": get_text("Color"),
            "Costo": get_num("Costo"),
            "Precio": get_num("Precio"),
            "Stock": int(get_num("Stock")),
            "Minimo": int(get_num("Minimo")),
            "SKU": get_text("SKU")
        }
        if not p["Categoria"] and "descripcion" in idx:
            desc = _normalize(str(row[idx["descripcion"]]))
            for w in ["pantalon","bermuda","botin","camisa","remera","campera","chaleco","buzo","zapatilla","guante","jean","gorra"]:
                if re.search(rf"\b{w}\b", desc):
                    p["Categoria"] = w
                    break

        # normalizar color por alias
        if p["Color"]:
            p["Color"] = _alias_color(p["Color"])
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

# === cantidad vs talle ===
_SIZE_TOKENS = {"xs","s","m","l","xl","xxl","xxxl"}

def _detect_talle(text_nrm: str, talles_cat: List[str]) -> Optional[str]:
    m = re.search(r"\b(?:talle|t)\s*([a-z0-9\-]+)\b", text_nrm)
    if m:
        return _normalize(m.group(1))
    tokens = text_nrm.split()
    for tok in tokens:
        t = _normalize(tok)
        if t in _SIZE_TOKENS or t in talles_cat:
            return t
    return None

def _infer_cantidad(text: str, talle: Optional[str], talles_cat: List[str]) -> int:
    t = _normalize(text)

    # palabras: "una/dos/..." priorizan cantidad
    for w, n in NUMBER_WORDS.items():
        if re.search(rf"\b{w}\b", t):
            return n

    # verbo + número → cantidad (si no es talle)
    m = re.search(r"\b(vendi|vendí|venta|compr[eé]|compre|compra|agrega|agreg[aá]|sumar|sum[aá]|descont[aá]|restar|resta|retir[eo])\s+(\d{1,3})\b", t)
    if m:
        cand = int(m.group(2))
        if not (talle and str(cand) == str(talle)):
            return cand

    nums = [int(x) for x in re.findall(r"\b(\d{1,3})\b", t)]
    if talle and re.fullmatch(r"\d{1,3}", str(talle)):
        nums = [n for n in nums if str(n) != str(talle)]
    return nums[0] if nums else 1

def _extract_entities_from_query(products: List[Dict[str,Any]], text: str) -> Dict[str, Optional[str]]:
    t = _normalize(text)
    tokens = [_singularize(tok) for tok in t.split()]
    t_sing = " ".join(tokens)

    cats_cat, prods_cat, colores_cat, talles_cat = _catalogo_valores(products)

    categoria = _pick_one_from(" " + t_sing + " ", cats_cat) or _detect_categoria_from_text(t_sing)
    producto = _pick_one_from(t_sing, prods_cat)
    color = _pick_one_from(t_sing, [*_SIZE_TOKENS, *colores_cat])  # evita confundir XS con color inexistente
    if color: color = _alias_color(color)

    talle = _detect_talle(t_sing, talles_cat)

    m2 = re.search(r"\bcod(?:igo)?\s*([a-z0-9\-]+)\b", t_sing)
    codigo = m2.group(1) if m2 else None

    return {"Categoria": categoria, "Producto": producto, "Talle": talle, "Color": color, "Codigo": codigo}

def _filtrar_por_campos(products: List[Dict[str,Any]], f: Dict[str, Optional[str]]) -> List[Dict[str,Any]]:
    def ok(p, k):
        v = _normalize(p.get(k,""))
        q = _normalize(f.get(k) or "")
        return (not q) or (q and q == v)
    if f.get("Codigo"):
        p = _find_by_code(f["Codigo"], products)
        return [p] if p else []
    return [p for p in products if ok(p,"Categoria") and ok(p,"Producto") and ok(p,"Talle") and ok(p,"Color")]

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

# ---------- Desempate inteligente ----------
def _score_match(prod: Dict[str,Any], filtro: Dict[str,Optional[str]]) -> int:
    score = 0
    if filtro.get("Categoria") and _normalize(prod.get("Categoria","")) == _normalize(filtro["Categoria"]): score += 3
    if filtro.get("Producto")  and _normalize(prod.get("Producto",""))  == _normalize(filtro["Producto"]): score += 2
    if filtro.get("Color")     and _normalize(prod.get("Color",""))     == _normalize(filtro["Color"]):    score += 2
    if filtro.get("Talle")     and _normalize(prod.get("Talle",""))     == _normalize(filtro["Talle"]):    score += 2
    return score

def _auto_pick_if_clear(candidatos: List[Dict[str,Any]], filtro: Dict[str,Optional[str]]) -> Optional[Dict[str,Any]]:
    if not candidatos: return None
    scores = [(p, _score_match(p, filtro)) for p in candidatos]
    scores.sort(key=lambda x: x[1], reverse=True)
    if len(scores) == 1:
        return scores[0][0]
    # si el primero supera por >=2 puntos al segundo, lo tomamos
    if scores[0][1] >= scores[1][1] + 2:
        return scores[0][0]
    return None

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

# ---- IA: parser de ITEMS múltiples (mejorado) ----
def _llm_parse_items(text: str) -> Optional[List[Dict[str,Any]]]:
    if not (OPENAI_API_KEY and OpenAI):
        return None
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        system = (
            "Sos un parser de español para control de stock. "
            "Devolvé JSON con este formato:\n"
            "{ \"items\": [ {\"accion\":\"agregar|descontar\",\"cantidad\":int,"
            "\"categoria\":str,\"producto\":str,\"talle\":str,\"color\":str,"
            "\"codigo\":str,\"precio_venta\":float|null} ] }\n"
            "- Partí el texto en ítems cuando haya comas o ' y '.\n"
            "- 'vender'/'vendí' => descontar; 'comprar'/'compré' => agregar.\n"
            "- cantidad por defecto = 1 si no está clara.\n"
            "- 'talle 44' o 't 44' es talle, no cantidad.\n"
            "- Si no sabés un campo, dejalo \"\" (vacío) o null.\n"
        )
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"system","content":system},{"role":"user","content":text}],
            response_format={"type":"json_object"},
            temperature=0
        )
        content = _limpiar_codeblock(resp.choices[0].message.content or "")
        data = json.loads(content)
        items = data.get("items")
        if isinstance(items, list) and items:
            return items
    except Exception as e:
        log.warning(f"LLM items parse off: {e}")
    return None

def _nlp_parse(text: str) -> Dict[str, Any]:
    t = _normalize(text)
    accion = _detectar_intencion(t)
    if accion == "anular_ultimo":
        return {"intent":"anular_ultimo", "data":{}}
    precio = _extraer_precio(t)
    return {"intent":"ajustar_stock","data":{"accion":accion,"query":text,"precio_venta":precio}}

# ============= MULTI-ITEM PARSER =============
def _split_into_items(text: str) -> List[str]:
    t = " " + _normalize(text) + " "
    t = re.sub(r"\s+y\s+", ",", t)
    partes = [p.strip() for p in t.split(",")]
    return [p for p in partes if p]

def _build_item_from_text(products, text: str, accion_global: Optional[str]) -> Dict[str, Any]:
    parsed = _nlp_parse(text)
    data = parsed.get("data", {})
    accion = data.get("accion") or accion_global or "descontar"

    filtro = _extract_entities_from_query(products, text)

    _, _, _, talles_cat = _catalogo_valores(products)
    cantidad = _infer_cantidad(text, filtro.get("Talle"), talles_cat)

    precio_venta = data.get("precio_venta")
    return {"accion":accion, "cantidad":cantidad, "precio_venta":precio_venta, "filtro":filtro, "texto":text}

# ----------------- MOVIMIENTOS / STOCK -----------------
def _update_stock(ws, prod, new_stock, idx):
    col = idx["stock"] + 1 if "stock" in idx else 8
    ws.update_cell(prod["row"], col, new_stock)

def _append_movement(ws_movs, tipo, prod, cantidad, precio_venta, costo_unit):
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ganancia = (precio_venta - costo_unit) * abs(int(cantidad)) if tipo == "salida" else 0
    ws_movs.append_row([
        fecha, tipo, prod.get("Codigo",""), prod.get("Producto",""),
        prod.get("Talle",""), prod.get("Color",""),
        prod.get("Categoria","") or "",
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
    candidatos = _filtrar_por_campos(products, filtro)
    if not candidatos:
        candidatos = _buscar_textual(products, " ".join([v for v in filtro.values() if v]))

    if not candidatos:
        slot = _siguiente_slot_faltante(filtro)
        if slot:
            await _preguntar_slot(update, slot, products)
            _set_pendiente(context, {"accion":accion,"cantidad":cantidad,"precio_venta":precio_venta,"filtro":filtro,"t":datetime.now().timestamp()})
            return
        await update.message.reply_text("No encontré coincidencias. Indicá *categoria, producto, talle y color*.", parse_mode="Markdown")
        return

    # Nuevo: intento de elección automática si hay empate
    if len(candidatos) > 1:
        elegido = _auto_pick_if_clear(candidatos, filtro)
        if elegido is None:
            lista = "\n".join([f"- {_descripcion_corta(p)} código: {p.get('Codigo') or 's/cod'}" for p in candidatos[:10]])
            await update.message.reply_text(
                f"Encontré varias coincidencias:\n{lista}\n\nDecime el *código* exacto o especificá mejor (p. ej. 'pantalon clasico 42 verde').",
                parse_mode="Markdown"
            )
            _set_pendiente(context, {"accion":accion,"cantidad":cantidad,"precio_venta":precio_venta,"filtro":filtro,"t":datetime.now().timestamp()})
            return
        else:
            candidatos = [elegido]

    # === ÚNICO PRODUCTO → AJUSTAR ===
    prod = candidatos[0]
    delta = abs(cantidad) if accion == "agregar" else -abs(cantidad)
    new_stock = max(0, prod["Stock"] + delta)
    _update_stock(ws_prod, prod, new_stock, idx)
    tipo = "entrada" if delta > 0 else "salida"
    precio_v = float(precio_venta or prod["Precio"] or 0)
    costo = float(prod["Costo"] or 0)
    _append_movement(ws_movs, tipo, prod, abs(delta), precio_v, costo)

    _set_pendiente(context, None)

    msg = f"✅ {('Sumé' if delta>0 else 'Desconté')} {abs(delta)} de {_descripcion_corta(prod)} (stock {prod['Stock']}→{new_stock})"
    if tipo == "salida" and precio_v and costo:
        msg += f"\nGanancia estimada: {_money((precio_v - costo)*abs(delta))}"
    await update.message.reply_text(msg)

# ----------------- UNDO ÚLTIMO MOVIMIENTO -----------------
def _get_headers_index(values: List[List[str]]) -> Dict[str,int]:
    headers = [h.strip().lower() for h in (values[0] if values else [])]
    return {h:i for i,h in enumerate(headers)}

def _find_product_by_rowdata(products, codigo, producto, talle, color, categoria) -> Optional[Dict[str,Any]]:
    if codigo:
        p = _find_by_code(codigo, products); 
        if p: return p
    f = {"Categoria":categoria or "", "Producto":producto or "", "Talle":talle or "", "Color":_alias_color(color or "")}
    cands = _filtrar_por_campos(products, f)
    if cands: return cands[0]
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
        last_row_num = len(values)
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
        categoria = get("descripcion").strip()

        products, pidx = _read_products(ws_prod)
        prod = _find_product_by_rowdata(products, codigo, producto, talle, color, categoria)
        if not prod:
            await update.message.reply_text("No pude ubicar el producto del último movimiento para revertir.")
            return

        try:
            cant = int(float((get("cantidad") or "0").replace(",", ".")))
        except:
            cant = 0
        if cant <= 0:
            await update.message.reply_text("El último movimiento no tiene cantidad válida para revertir.")
            return

        delta = -abs(cant) if tipo == "entrada" else +abs(cant)
        new_stock = max(0, prod["Stock"] + delta)
        _update_stock(ws_prod, prod, new_stock, pidx)
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
            if has_categoria:
                partes = [partes[0], ""] + partes[1:]
        else:
            if not has_categoria:
                partes = [partes[0]] + partes[2:]

        def to_float(x): 
            try: return float(str(x).replace(",", "."))
            except: return 0.0
        def to_int(x):
            try: return int(float(str(x).replace(",", ".")))
            except: return 0

        if has_categoria and len(partes) != 10: raise ValueError("Se esperaban 10 campos (incluida Categoria).")
        if not has_categoria and len(partes) != 9: raise ValueError("Se esperaban 9 campos (sin Categoria).")

        if has_categoria:
            partes[5] = to_float(partes[5]); partes[6] = to_float(partes[6])
            partes[7] = to_int(partes[7]);   partes[8] = to_int(partes[8])
        else:
            partes[5] = to_float(partes[5]); partes[6] = to_float(partes[6])
            partes[7] = to_int(partes[7]);   partes[8] = to_int(partes[8])

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

    if any(w in text_nrm for w in UNDO_WORDS):
        await _cmd_anular(update, context)
        return

    sh, ws_prod, ws_movs = _open_sheet()
    products, idx = _read_products(ws_prod)

    # IA: primero intento parsear todo en items con LLM
    items_llm = _llm_parse_items(text)
    if items_llm:
        ajustes: List[Dict[str,Any]] = []
        intent_global = _detectar_intencion(text_nrm)
        for it in items_llm:
            accion = (it.get("accion") or intent_global or "descontar").strip() if isinstance(it.get("accion"), str) else (intent_global or "descontar")
            precio_venta = it.get("precio_venta")
            # construir filtro combinando IA + heurística
            raw = " ".join([str(x) for x in [it.get("categoria",""), it.get("producto",""), it.get("talle",""), it.get("color",""), it.get("codigo","")] if x])
            fil = _extract_entities_from_query(products, raw or text)
            # refinar con campos IA si los trajo
            if it.get("categoria"): fil["Categoria"] = _normalize(it["categoria"])
            if it.get("producto"):  fil["Producto"]  = _normalize(it["producto"])
            if it.get("talle"):     fil["Talle"]     = _normalize(it["talle"])
            if it.get("color"):     fil["Color"]     = _alias_color(it["color"])
            if it.get("codigo"):    fil["Codigo"]    = _normalize(it["codigo"])
            # cantidad
            _, _, _, talles_cat = _catalogo_valores(products)
            cantidad = int(it.get("cantidad") or _infer_cantidad(text, fil.get("Talle"), talles_cat) or 1)
            ajustes.append({"accion":accion,"cantidad":cantidad,"precio_venta":precio_venta,"filtro":fil,"texto":text})
        await update.message.reply_text(f"🧾 Detecté {len(ajustes)} ítems. Voy aplicando…")
        for adj in ajustes:
            await _resolver_y_ajustar(update, context, adj["accion"], adj["cantidad"], adj["precio_venta"], adj["filtro"], products, idx, ws_prod, ws_movs)
        return

    # Si no hay IA o falló, uso el parser local (comas/“ y ”)
    intent_global = _detectar_intencion(text_nrm)
    parts = _split_into_items(text)
    if len(parts) <= 1:
        parsed = _nlp_parse(text)
        intent = parsed.get("intent")
        data = parsed.get("data", {})

        if intent in ("reporte","faltantes","ganancias"):
            if intent == "reporte": await _cmd_reporte(update, context)
            elif intent == "faltantes": await _cmd_faltantes(update, context)
            else: await _cmd_ganancias(update, context)
            return

        accion = data.get("accion") or ("agregar" if re.search(r"\bcompr|agreg|sum", text_nrm) else "descontar")
        precio_venta = data.get("precio_venta")

        filtro = _extract_entities_from_query(products, text)
        _, _, _, talles_cat = _catalogo_valores(products)
        cantidad = _infer_cantidad(text, filtro.get("Talle"), talles_cat)

        await _resolver_y_ajustar(update, context, accion, cantidad, precio_venta, filtro, products, idx, ws_prod, ws_movs)
        return

    ajustes: List[Dict[str,Any]] = []
    for it in parts:
        parsed = _build_item_from_text(products, it, intent_global if intent_global in ("agregar","descontar") else None)
        ajustes.append(parsed)

    await update.message.reply_text(f"🧾 Detecté {len(ajustes)} ítems. Voy aplicando…")
    for adj in ajustes:
        await _resolver_y_ajustar(update, context, adj["accion"], adj["cantidad"], adj["precio_venta"], adj["filtro"], products, idx, ws_prod, ws_movs)

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
