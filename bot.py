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

# Cabeceras "amigables"; varias son opcionales y el bot se adapta
PRODUCTS_HEADERS = [
    "Codigo", "Categoria", "Modelo", "Producto", "Talle", "Color",
    "Medida", "Tipo", "Costo", "Precio", "Stock", "Minimo", "SKU"
]
MOVS_HEADERS = [
    "Fecha", "Tipo", "Codigo", "Producto", "Talle", "Color", "Descripcion",
    "Cantidad", "PrecioVenta", "CostoUnitario", "Ganancia"
]

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
    "shorts":"bermuda", "short":"bermuda", "pares":"par", "unidades":"unidad"
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

def _numbers(text_nrm: str) -> List[int]:
    return [int(n) for n in re.findall(r"\b(\d{1,4})\b", text_nrm)]

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

def _get(ws) -> Tuple[List[List[str]], Dict[str,int]]:
    values = ws.get_all_values()
    headers = [h.strip() for h in (values[0] if values else [])]
    idx = {h.lower(): i for i, h in enumerate(headers)}
    return values, idx

def _read_products(ws):
    values, idx = _get(ws)
    if not values:
        ws.append_row(PRODUCTS_HEADERS)
        return [], {}
    headers = [h.strip() for h in values[0]]
    idx = {h.lower(): i for i, h in enumerate(headers)}
    rows = values[1:]
    products = []
    for r_i, row in enumerate(rows, start=2):
        if not any(row): continue

        def get_text(key):
            k = key.lower()
            return str(row[idx[k]]).strip() if k in idx and idx[k] < len(row) else ""

        def get_num(key):
            k = key.lower()
            try: return float(str(row[idx[k]]).replace(",", ".") or 0)
            except: return 0

        p = {
            "row": r_i,
            "Codigo": get_text("Codigo"),
            "Categoria": get_text("Categoria") if "categoria" in idx else "",
            "Modelo": get_text("Modelo") if "modelo" in idx else "",
            "Producto": get_text("Producto") if "producto" in idx else "",
            "Talle": get_text("Talle") if "talle" in idx else "",
            "Color": get_text("Color") if "color" in idx else "",
            "Medida": get_text("Medida") if "medida" in idx else "",
            "Tipo": get_text("Tipo") if "tipo" in idx else "",
            "Costo": get_num("Costo") if "costo" in idx else 0,
            "Precio": get_num("Precio") if "precio" in idx else 0,
            "Stock": int(get_num("Stock") if "stock" in idx else 0),
            "Minimo": int(get_num("Minimo") if "minimo" in idx else 0),
            "SKU": get_text("SKU") if "sku" in idx else "",
        }

        # Compat: si no hay Categoria pero existe Descripcion, intentar inferir
        if not p["Categoria"] and "descripcion" in idx:
            desc = _normalize(str(row[idx["descripcion"]]))
            for w in ["pantalon","bermuda","botin","zapatilla","camisa","remera","campera","chaleco","buzo","guante","jean","gorra","matafuego"]:
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

def _catalogo_valores(products: List[Dict[str,Any]]):
    # catálogos para detección por texto
    categorias = sorted({ _normalize(p.get("Categoria","")) for p in products if p.get("Categoria") }, key=len, reverse=True)
    modelos    = sorted({ _normalize(p.get("Modelo","")) for p in products if p.get("Modelo") }, key=len, reverse=True)
    prods      = sorted({ _normalize(p.get("Producto","")) for p in products if p.get("Producto") }, key=len, reverse=True)
    colores    = sorted({ _normalize(p.get("Color","")) for p in products if p.get("Color") }, key=len, reverse=True)
    talles     = sorted({ _normalize(p.get("Talle","")) for p in products if p.get("Talle") }, key=len, reverse=True)
    medidas    = sorted({ _normalize(p.get("Medida","")) for p in products if p.get("Medida") }, key=len, reverse=True)
    tipos      = sorted({ _normalize(p.get("Tipo","")) for p in products if p.get("Tipo") }, key=len, reverse=True)
    return categorias, modelos, prods, colores, talles, medidas, tipos

def _pick_one_from(text_nrm: str, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c and c in text_nrm:
            return c
    return None

def _detect_categoria_from_text(text_nrm: str) -> Optional[str]:
    for w in ["pantalon","bermuda","botin","zapatilla","camisa","remera","campera","chaleco","buzo","guante","jean","gorra","matafuego"]:
        if re.search(rf"\b{w}s?\b", text_nrm):
            return w
    return None

def _extract_entities_from_query(products: List[Dict[str,Any]], text: str) -> Dict[str, Optional[str]]:
    t = _normalize(text)
    tokens = [_singularize(tok) for tok in t.split()]
    t_sing = " ".join(tokens)

    cats_cat, modelos_cat, prods_cat, colores_cat, talles_cat, medidas_cat, tipos_cat = _catalogo_valores(products)

    categoria = _pick_one_from(" " + t_sing + " ", cats_cat) or _detect_categoria_from_text(t_sing)
    modelo    = _pick_one_from(t_sing, modelos_cat)
    producto  = _pick_one_from(t_sing, prods_cat)
    color     = _pick_one_from(t_sing, colores_cat)
    tipo      = _pick_one_from(t_sing, tipos_cat)

    # Talle: por palabra clave o por pertenencia al catálogo de talles
    talle = None
    m = re.search(r"\b(?:talle|t|n°|nº|numero|nro)\s*([a-z0-9\-]+)\b", t_sing)
    if m:
        talle = m.group(1)
    else:
        for tok in tokens:
            if tok in talles_cat:
                talle = tok; break
            if re.fullmatch(r"\d{2}", tok) and tok in talles_cat:
                talle = tok; break

    # Medida (matafuegos): intenta 3kg/5kg/10kg/50kg, etc
    medida = None
    m2 = re.search(r"\b(\d{1,3})\s*(kg|lts?|l)\b", t_sing)
    if m2:
        medida = f"{m2.group(1)}{m2.group(2)}"
    elif medidas_cat:
        for tok in tokens:
            if tok in medidas_cat:
                medida = tok; break

    # Código explícito
    m3 = re.search(r"\bcod(?:igo)?\s*([a-z0-9\-]+)\b", t_sing)
    codigo = m3.group(1) if m3 else None

    return {
        "Categoria": categoria, "Modelo": modelo, "Producto": producto,
        "Talle": talle, "Color": color, "Medida": medida, "Tipo": tipo, "Codigo": codigo
    }

def _filtrar_por_campos(products: List[Dict[str,Any]], f: Dict[str, Optional[str]]) -> List[Dict[str,Any]]:
    def ok(p, k):
        q = _normalize(f.get(k) or "")
        if not q:  # no filtrar por campo ausente
            return True
        v = _normalize(p.get(k, "") or "")
        if not v:  # la fila no tiene ese dato → no descarto por esto
            return True
        return q == v

    if f.get("Codigo"):
        p = _find_by_code(f["Codigo"], products)
        return [p] if p else []

    return [
        p for p in products
        if ok(p,"Categoria") and ok(p,"Modelo") and ok(p,"Producto")
        and ok(p,"Talle") and ok(p,"Color") and ok(p,"Medida") and ok(p,"Tipo")
    ]

def _buscar_textual(products: List[Dict[str,Any]], query: str) -> List[Dict[str,Any]]:
    q = _normalize(query)
    toks = set(q.split())
    out = []
    for p in products:
        texto = _normalize(" ".join([
            p.get("Codigo",""), p.get("Categoria",""), p.get("Modelo",""), p.get("Producto",""),
            p.get("Talle",""), p.get("Color",""), p.get("Medida",""), p.get("Tipo",""), p.get("SKU","")
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

# ======== DESAMBIGUACIÓN CANTIDAD vs TALLE =========
def _cantidad_desde_verbos(text_nrm: str) -> Optional[int]:
    # Busca número inmediatamente después de verbos típicos
    m = re.search(r"\b(vendi|vendí|venta|compr(e|é)|compra|agrega|agregá|sum(a|á)|entrada|salida|desconta|descontá)\D+?(\d{1,4})\b", text_nrm)
    if m:
        try: return int(m.group(3))
        except: return None
    return None

def _marcas_talle(text_nrm: str) -> Optional[str]:
    m = re.search(r"\b(?:talle|t|n°|nº|numero|nro)\s*([a-z0-9\-]+)\b", text_nrm)
    return m.group(1) if m else None

def _separar_cantidad_talle(text_nrm: str, talles_catalogo: set) -> Tuple[int, Optional[str]]:
    # 1) Si hay marca explícita de talle → eso es TALLE
    talle_marcado = _marcas_talle(text_nrm)
    if talle_marcado:
        cant = _cantidad_desde_verbos(text_nrm)
        if cant: return cant, talle_marcado
        # si no hay cantidad por verbos, buscar otro número que NO sea el talle
        nums = _numbers(text_nrm)
        otros = [n for n in nums if str(n) != str(talle_marcado)]
        return (otros[0] if otros else 1), talle_marcado

    # 2) Si aparece un número que pertenece al catálogo de talles → ese es TALLE
    nums = _numbers(text_nrm)
    if nums:
        # ¿hay cantidad marcada por verbos?
        cant_v = _cantidad_desde_verbos(text_nrm)
        if cant_v is not None:
            # si además hay un número que es talle de catálogo, usamos ambos
            for n in nums:
                if str(n) in talles_catalogo:
                    return cant_v, str(n)
            # si no hay talle claro → no fijamos talle aquí
            return cant_v, None
        # sin cantidad marcada: si hay número del catálogo → es talle; cantidad = 1
        for n in nums:
            if str(n) in talles_catalogo:
                return 1, str(n)

    # 3) Default
    return 1, None

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

    precio = _extraer_precio(t)
    base = {"intent":"ajustar_stock","data":{"accion":accion,"query":text,"precio_venta":precio}}

    if OPENAI_API_KEY and OpenAI:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            system = (
                "Sos un parser de español para control de stock. "
                "Devolvé JSON con 'intent' y 'data'. intents: 'ajustar_stock','reporte','faltantes','ganancias','anular_ultimo'. "
                "Para 'ajustar_stock' incluí: {'accion':'agregar'|'descontar','cantidad':int|null,'query':str,'precio_venta':float|null, "
                "'categoria':str,'modelo':str,'producto':str,'talle':str,'color':str,'medida':str,'tipo':str,'codigo':str}."
            )
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":system},{"role":"user","content":text}],
                response_format={"type":"json_object"},
                temperature=0
            )
            content = _limpiar_codeblock(resp.choices[0].message.content or "")
            data = json.loads(content)
            if not data.get("intent"): data["intent"] = "ajustar_stock"
            return data
        except Exception as e:
            log.warning(f"OpenAI deshabilitado por error: {e}")

    return base

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
    precio_venta = data.get("precio_venta")

    # entidades
    filtro = _extract_entities_from_query(products, text)
    # si IA trajo campos, priorizalos
    for k_src, k_dst in [
        ("categoria","Categoria"), ("modelo","Modelo"), ("producto","Producto"),
        ("talle","Talle"), ("color","Color"), ("medida","Medida"), ("tipo","Tipo"),
        ("codigo","Codigo")
    ]:
        v = data.get(k_src)
        if v and not filtro.get(k_dst):
            filtro[k_dst] = _normalize(v)

    # cantidad se define luego con catálogo de talles en _resolver_y_ajustar
    return {"accion":accion, "cantidad":None, "precio_venta":precio_venta, "filtro":filtro, "texto":text}

# ----------------- MOVIMIENTOS / STOCK -----------------
def _update_stock(ws, prod, new_stock, idx):
    col = idx["stock"] + 1 if "stock" in idx else 11  # fallback a col 'Stock'
    ws.update_cell(prod["row"], col, new_stock)

def _append_movement(ws_movs, tipo, prod, cantidad, precio_venta, costo_unit):
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ganancia = (precio_venta - costo_unit) * abs(int(cantidad)) if tipo == "salida" else 0
    ws_movs.append_row([
        fecha, tipo, prod.get("Codigo",""),
        (prod.get("Modelo") or prod.get("Producto") or "").strip(),
        prod.get("Talle",""), prod.get("Color",""),
        (prod.get("Categoria","") or prod.get("Medida","") or prod.get("Tipo","") or "").strip(),
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
    partes = [
        p.get("Categoria",""), p.get("Modelo","") or p.get("Producto",""),
        p.get("Talle",""), p.get("Color","") or p.get("Medida","") or p.get("Tipo","")
    ]
    return " ".join([x for x in partes if x]).strip()

# ----------------- SLOT-FILLING -----------------
def _estado_pendiente(context: ContextTypes.DEFAULT_TYPE) -> Optional[Dict[str,Any]]:
    return context.user_data.get("pending_adjust")

def _set_pendiente(context: ContextTypes.DEFAULT_TYPE, data: Optional[Dict[str,Any]]):
    if data is None:
        context.user_data.pop("pending_adjust", None)
    else:
        context.user_data["pending_adjust"] = data

def _siguiente_slot_faltante(filtro: Dict[str, Optional[str]]) -> Optional[str]:
    for k in ["Categoria","Modelo","Producto","Talle","Color"]:
        if not filtro.get(k):
            return k
    return None

async def _preguntar_slot(update: Update, slot: str, products: List[Dict[str,Any]]):
    if slot == "Categoria":
        await update.message.reply_text("¿Es *pantalon*, *botin*, *zapatilla*, *camisa*, *matafuego*…?", parse_mode="Markdown")
    elif slot == "Modelo":
        await update.message.reply_text("¿Qué *modelo*? (ej. 'clasico', 'cargo', 'frances', 'ripstop')", parse_mode="Markdown")
    elif slot == "Producto":
        await update.message.reply_text("¿Qué *línea/producto*? (podés dejarlo vacío)", parse_mode="Markdown")
    elif slot == "Talle":
        await update.message.reply_text("¿Qué *talle*? (ej. 42)", parse_mode="Markdown")
    elif slot == "Color":
        await update.message.reply_text("¿Qué *color*?", parse_mode="Markdown")
    else:
        await update.message.reply_text("Necesito un dato más…")

# ----------- Núcleo: resolver + desambiguar cantidad/talle -----------
async def _resolver_y_ajustar(update, context, accion, cantidad, precio_venta, filtro, products, idx, ws_prod, ws_movs, texto_original=None):
    # catálogo de talles reales
    _, _, _, _, talles_cat, _, _ = _catalogo_valores(products)
    talles_set = set(talles_cat)

    # Si no vino cantidad definida aún, calcular con desambiguación usando el texto original si está
    text_nrm = _normalize(texto_original or " ".join([str(x) for x in filtro.values() if x]))
    cant_calc, talle_calc = _separar_cantidad_talle(text_nrm, talles_set)

    if cantidad is None:
        cantidad = cant_calc
    # Si no vino talle en filtro y el parser detectó talle válido → asignarlo
    if not (filtro.get("Talle")) and talle_calc:
        filtro["Talle"] = talle_calc

    # 1) Intento por filtro exacto
    candidatos = _filtrar_por_campos(products, filtro)
    if not candidatos:
        # 2) Intento textual con campos presentes
        q_parts = [str(filtro[k]) for k in ("Categoria","Modelo","Producto","Talle","Color","Medida","Tipo","Codigo") if filtro.get(k)]
        candidatos = _buscar_textual(products, " ".join(q_parts))

    if not candidatos:
        slot = _siguiente_slot_faltante(filtro)
        if slot:
            await _preguntar_slot(update, slot, products)
            _set_pendiente(context, {
                "accion":accion,"cantidad":cantidad,"precio_venta":precio_venta,
                "filtro":filtro,"t":datetime.now().timestamp()
            })
            return
        await update.message.reply_text("No encontré coincidencias. Indicá *categoria, modelo/producto, talle y color*.", parse_mode="Markdown")
        return

    if len(candidatos) > 1:
        lista = "\n".join([f"- {_descripcion_corta(p)} código: {p.get('Codigo') or 's/cod'}" for p in candidatos[:10]])
        await update.message.reply_text(
            f"Encontré varias coincidencias:\n{lista}\n\nDecime el *código* exacto o especificá mejor (p. ej. 'pantalon clasico 42 verde').",
            parse_mode="Markdown"
        )
        _set_pendiente(context, {
            "accion":accion,"cantidad":cantidad,"precio_venta":precio_venta,
            "filtro":filtro,"t":datetime.now().timestamp()
        })
        return

    # === ÚNICO PRODUCTO → AJUSTAR ===
    prod = candidatos[0]
    delta = abs(int(cantidad)) if accion == "agregar" else -abs(int(cantidad))
    new_stock = max(0, int(prod["Stock"]) + delta)
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
        p = _find_by_code(codigo, products)
        if p: return p
    f = {"Categoria":categoria or "", "Producto":producto or "", "Talle":talle or "", "Color":color or ""}
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

        tipo = _normalize(get("tipo"))
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
            cant = int(float(get("cantidad").replace(",", ".") or "0"))
        except:
            cant = 0
        if cant <= 0:
            await update.message.reply_text("El último movimiento no tiene cantidad válida para revertir.")
            return

        delta = -abs(cant) if tipo == "entrada" else +abs(cant)
        new_stock = max(0, int(prod["Stock"]) + delta)
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
        "• 'Vendí 2 botines francés 42 negros a $34000'\n"
        "• 'Código 79 vendí 1'\n"
        "• *Múltiples*: 'vendí 2 pantalones clásico 44, una camisa 42 y una camisa 44'\n"
        "• 'Mostrame los faltantes' | 'Ganancias' | 'anular'\n\n"
        "Tip: el bot usa tu catálogo real de talles para no confundir *42* (talle) con cantidad."
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

# /agregar: alta de fila completa compatible con columnas opcionales
async def _cmd_agregar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # /agregar codigo,categoria,modelo,producto,talle,color,medida,tipo,costo,precio,stock,minimo,sku
        texto = " ".join(context.args)
        partes = [p.strip() for p in texto.split(",")]

        _, ws_prod, _ = _open_sheet()
        values, idx = _get(ws_prod)
        headers = [h.strip().lower() for h in values[0]] if values else []

        # Mapeo dinámico según existan columnas
        fields = ["codigo","categoria","modelo","producto","talle","color","medida","tipo","costo","precio","stock","minimo","sku"]
        have = [h for h in fields if h in headers]
        need = len(have)

        if len(partes) != need:
            raise ValueError(f"Se esperaban {need} campos en este orden: {', '.join(have)}")

        # Normalizar tipos numéricos si existen
        def to_float(x): 
            try: return float(str(x).replace(",", "."))
            except: return 0.0
        def to_int(x):
            try: return int(float(str(x).replace(",", ".")))
            except: return 0

        for pos, name in enumerate(have):
            if name in ("costo","precio"):
                partes[pos] = to_float(partes[pos])
            if name in ("stock","minimo"):
                partes[pos] = to_int(partes[pos])

        ws_prod.append_row(partes, value_input_option="USER_ENTERED")
        await update.message.reply_text("✅ Producto agregado correctamente.")
    except Exception as e:
        log.error(f"/agregar error: {e}")
        await update.message.reply_text(
            "❌ Formato:\n/agregar codigo,categoria,modelo,producto,talle,color,medida,tipo,costo,precio,stock,minimo,sku\n"
            "El orden exacto depende de las columnas existentes en tu hoja."
        )

# ----------------- MENSAJES -----------------
async def _on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text or ""
    text_nrm = _normalize(text)

    # Atajo: anular por texto
    if any(w in text_nrm for w in UNDO_WORDS):
        await _cmd_anular(update, context)
        return

    sh, ws_prod, ws_movs = _open_sheet()
    products, idx = _read_products(ws_prod)

    intent_global = _detectar_intencion(text_nrm)
    if intent_global == "anular_ultimo":
        await _cmd_anular(update, context); return

    items = _split_into_items(text)
    if len(items) <= 1:
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
        precio_venta = data.get("precio_venta")

        filtro = _extract_entities_from_query(products, text)
        for k_src, k_dst in [
            ("categoria","Categoria"), ("modelo","Modelo"), ("producto","Producto"),
            ("talle","Talle"), ("color","Color"), ("medida","Medida"), ("tipo","Tipo"),
            ("codigo","Codigo")
        ]:
            v = data.get(k_src)
            if v and not filtro.get(k_dst):
                filtro[k_dst] = _normalize(v)

        # cantidad y talle se desambiguarán dentro de _resolver_y_ajustar
        await _resolver_y_ajustar(update, context, accion, None, precio_venta, filtro, products, idx, ws_prod, ws_movs, texto_original=text)
        return

    # Multi-ítem
    ajustes: List[Dict[str,Any]] = []
    for it in items:
        adj = _build_item_from_text(products, it, intent_global if intent_global in ("agregar","descontar") else None)
        ajustes.append(adj)

    await update.message.reply_text(f"🧾 Detecté {len(ajustes)} ítems. Voy aplicando…")
    for adj in ajustes:
        await _resolver_y_ajustar(
            update, context,
            adj["accion"], adj["cantidad"], adj["precio_venta"],
            adj["filtro"], products, idx, ws_prod, ws_movs, texto_original=adj["texto"]
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
