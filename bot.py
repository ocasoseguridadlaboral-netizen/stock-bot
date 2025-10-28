# bot.py
import os, re, json, logging, sys, unicodedata
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
import pkg_resources

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ==== IA (OpenAI) opcional ====
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ----------------- LOAD ENV -----------------
load_dotenv()

TELEGRAM_TOKEN = (os.getenv("TELEGRAM_TOKEN") or "").strip()
GOOGLE_SHEET_ID = (os.getenv("GOOGLE_SHEET_ID") or "").strip()
OPENAI_API_KEY = (os.getenv("OPENAI_API_KEY") or "").strip()
OPENAI_MODEL = (os.getenv("OPENAI_MODEL") or "gpt-4o-mini").strip()
GOOGLE_SERVICE_ACCOUNT_JSON = (os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or "").strip()

if not TELEGRAM_TOKEN or not GOOGLE_SHEET_ID or not GOOGLE_SERVICE_ACCOUNT_JSON:
    raise RuntimeError("Faltan variables de entorno obligatorias: TELEGRAM_TOKEN, GOOGLE_SHEET_ID, GOOGLE_SERVICE_ACCOUNT_JSON")

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
log = logging.getLogger("stock-bot")

# ----------------- HEADERS / SHEETS -----------------
PRODUCTS_HEADERS = ["Codigo", "Categoria", "Producto", "Talle", "Color", "Costo", "Precio", "Stock", "Minimo", "SKU"]
MOVS_HEADERS = ["Fecha", "Tipo", "Codigo", "Categoria", "Producto", "Talle", "Color", "Cantidad", "PrecioVenta", "CostoUnitario", "Ganancia", "Nota"]

def _gs_client():
    info = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
    creds = Credentials.from_service_account_info(info, scopes=[
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive"
    ])
    return gspread.authorize(creds)

def _open_sheet():
    gc = _gs_client()
    sh = gc.open_by_key(GOOGLE_SHEET_ID)
    try:
        ws_prod = sh.worksheet("Productos")
    except gspread.WorksheetNotFound:
        ws_prod = sh.add_worksheet("Productos", rows=2000, cols=len(PRODUCTS_HEADERS))
        ws_prod.append_row(PRODUCTS_HEADERS)
    try:
        ws_movs = sh.worksheet("Movimientos")
    except gspread.WorksheetNotFound:
        ws_movs = sh.add_worksheet("Movimientos", rows=2000, cols=len(MOVS_HEADERS))
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
        def get_text(col): return str(row[idx[col]]).strip() if col in idx and idx[col] < len(row) else ""
        def get_num(col):
            try: return float(str(row[idx[col]]).replace(",", ".") or 0)
            except: return 0
        p = {
            "row": r_i,
            "Codigo": get_text("codigo"),
            "Categoria": get_text("categoria"),
            "Producto": get_text("producto"),
            "Talle": get_text("talle"),
            "Color": get_text("color"),
            "Costo": get_num("costo"),
            "Precio": get_num("precio"),
            "Stock": int(get_num("stock")),
            "Minimo": int(get_num("minimo")),
            "SKU": get_text("sku")
        }
        if p["Codigo"] or p["Producto"] or p["Categoria"]:
            products.append(p)
    return products, idx

# ----------------- NORMALIZACIÓN -----------------
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
    "gorras":"gorra", "camperas":"campera", "buzos":"buzo",
    "bermudas":"bermuda", "anteojos":"anteojo", "guantes":"guante", "chalecos":"chaleco"
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

# ----------------- CATÁLOGOS / BÚSQUEDA -----------------
def _catalogo_valores(products: List[Dict[str,Any]]) -> Tuple[List[str], List[str], List[str], List[str]]:
    cats = sorted({ _normalize(p.get("Categoria","")) for p in products if p.get("Categoria") }, key=len, reverse=True)
    prods = sorted({ _normalize(p.get("Producto","")) for p in products if p.get("Producto") }, key=len, reverse=True)
    colores = sorted({ _normalize(p.get("Color","")) for p in products if p.get("Color") }, key=len, reverse=True)
    talles = sorted({ _normalize(p.get("Talle","")) for p in products if p.get("Talle") }, key=len, reverse=True)
    return cats, prods, colores, talles

def _pick_one_from(text_nrm: str, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c and c in text_nrm:
            return c
    return None

def _find_by_code(code: str, products: List[Dict[str,Any]]) -> Optional[Dict[str,Any]]:
    c = _normalize(code)
    for p in products:
        if _normalize(p.get("Codigo","")) == c or _normalize(p.get("SKU","")) == c:
            return p
    return None

def _filtrar_por_campos(products: List[Dict[str,Any]], f: Dict[str, Optional[str]]) -> List[Dict[str,Any]]:
    def ok(p, k):
        v = _normalize(p.get(k,""))
        q = _normalize(f.get(k) or "")
        return (not q) or (q and q == v)
    return [p for p in products if ok(p,"Categoria") and ok(p,"Producto") and ok(p,"Color") and ok(p,"Talle")]

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

# ----------------- INTENCIÓN / CANTIDAD vs TALLE -----------------
VERBOS_SUMA = {"compre","compré","compra","agrega","agregá","agregar","sumar","sumá","suma","entrada","entraron","ingresa","ingresá","agrego","ingreso"}
VERBOS_RESTA = {"vendi","vendí","venta","vendimos","desconta","descontá","descontar","restar","resta","salida","salieron","retiro","retiré","descuento"}

def _detectar_intencion(text_nrm: str) -> Optional[str]:
    tokens = text_nrm.split()
    if any(v in tokens for v in VERBOS_RESTA): return "descontar"
    if any(v in tokens for v in VERBOS_SUMA): return "agregar"
    if re.search(r"\bvend", text_nrm): return "descontar"
    if re.search(r"\bcompr", text_nrm): return "agregar"
    return None

def _extraer_cantidad_desde_patrones(text_nrm: str) -> Optional[int]:
    # Verbos + número
    m = re.search(r"\b(?:vend[ií]|compr[eé]|agreg[aoá]|sum[eaó]|rest[aeó]|descont[aeó])\s+(\d{1,4})\b", text_nrm)
    if m:
        try: v = int(m.group(1));  return v if v>0 else None
        except: pass
    # x2
    m = re.search(r"\bx\s*(\d{1,4})\b", text_nrm)
    if m:
        try: v = int(m.group(1));  return v if v>0 else None
        except: pass
    # 2u / 2 uds / 2 unidades
    m = re.search(r"\b(\d{1,4})\s*(?:u|uds?|unidades?)\b", text_nrm)
    if m:
        try: v = int(m.group(1));  return v if v>0 else None
        except: pass
    # 2 + sustantivo
    m = re.search(r"\b(\d{1,4})\s+(?:pantalones?|bermudas?|botines?|zapatillas?|camisas?|remeras?|camperas?|chalecos?|buzos?|guantes?|jeans?|gorras?|anteojos?)\b", text_nrm)
    if m:
        try: v = int(m.group(1));  return v if v>0 else None
        except: pass
    return None

def _resolver_cantidad_y_talle(text: str, talles_cat: List[str]) -> Tuple[Optional[int], Optional[str]]:
    t = _normalize(text)
    # Talle explícito
    m_talle = re.search(r"\b(?:talle|t)\s*([a-z0-9\-]+)\b", t)
    talle = m_talle.group(1) if m_talle else None
    # Cantidad por patrones
    cantidad = _extraer_cantidad_desde_patrones(t)
    # Números sueltos para talle
    if not talle:
        nums = re.findall(r"\b(\d{1,3})\b", t)
        nums_filtrados = [n for n in nums if not (cantidad is not None and n == str(cantidad))]
        for n in nums_filtrados:
            if n in talles_cat:
                talle = n
                break
    if cantidad is None:
        cantidad = 1
    return cantidad, talle

def _detect_categoria_from_text(t_sing: str) -> Optional[str]:
    # categorías comunes
    cats = ["pantalon","bermuda","camisa","remera","botin","zapatilla","campera","chaleco","buzo","guante","gorra","anteojo","jean"]
    for c in cats:
        if re.search(rf"\b{c}s?\b", t_sing):
            return c
    return None

def _extract_entities_from_query(products: List[Dict[str,Any]], text: str) -> Dict[str, Optional[str]]:
    t = _normalize(text)
    tokens = [_singularize(tok) for tok in t.split()]
    t_sing = " ".join(tokens)

    cats_cat, prods_cat, colores_cat, talles_cat = _catalogo_valores(products)

    categoria = _pick_one_from(" "+t_sing+" ", cats_cat) or _detect_categoria_from_text(t_sing)
    producto  = _pick_one_from(t_sing, prods_cat)
    color     = _pick_one_from(t_sing, colores_cat)

    m2 = re.search(r"\bcod(?:igo)?\s*([a-z0-9\-]+)\b", t_sing)
    codigo = m2.group(1) if m2 else None

    cantidad, talle = _resolver_cantidad_y_talle(text, talles_cat)

    return {
        "Categoria": categoria,
        "Producto": producto,
        "Talle": talle,
        "Color": color,
        "Codigo": codigo,
        "_CantidadHint": cantidad
    }

# ----------------- IA OPCIONAL PARA NLU -----------------
def _nlp_parse(text: str) -> Dict[str, Any]:
    t = _normalize(text)
    accion = _detectar_intencion(t)
    base = {"intent":"ajustar_stock","data":{"accion":accion,"query":text}}

    if OPENAI_API_KEY and OpenAI:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            system = (
                "Sos un parser de español para control de stock. "
                "Devolvé JSON con 'intent' y 'items'. "
                "intent: 'ajustar_stock' | 'reporte' | 'faltantes' | 'ganancias' | 'anular'. "
                "'items' es una lista de objetos con: {categoria, producto, talle, color, codigo, cantidad, precio_venta?}."
                "Si un campo no está, dejalo vacío o null. Nunca inventes SKU/códigos."
            )
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":system},{"role":"user","content":text}],
                response_format={"type":"json_object"},
                temperature=0
            )
            data = json.loads(resp.choices[0].message.content)
            if "intent" not in data:
                data["intent"] = base["intent"]
            return data
        except Exception as e:
            log.warning(f"OpenAI deshabilitado por error: {e}")

    return base

# ----------------- MOVIMIENTOS / STOCK -----------------
def _update_stock(ws, prod, new_stock, idx):
    col = (idx.get("stock") or 7) + 1  # H=Stock (0-based a 7)
    ws.update_cell(prod["row"], col, new_stock)

def _append_movement(ws_movs, tipo, prod, cantidad, precio_venta, costo_unit, nota=""):
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ganancia = (float(precio_venta or 0) - float(costo_unit or 0)) * abs(int(cantidad)) if tipo == "salida" else 0
    ws_movs.append_row([
        fecha, tipo, prod.get("Codigo",""), prod.get("Categoria",""), prod.get("Producto",""),
        prod.get("Talle",""), prod.get("Color",""), int(cantidad), float(precio_venta or 0),
        float(costo_unit or 0), float(ganancia), nota
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

# ----------------- ÚLTIMA OPERACIÓN (para anular) -----------------
def _set_last_operation(context: ContextTypes.DEFAULT_TYPE, op: Optional[Dict[str,Any]]):
    if op is None:
        context.user_data.pop("last_op", None)
    else:
        context.user_data["last_op"] = op

def _get_last_operation(context: ContextTypes.DEFAULT_TYPE) -> Optional[Dict[str,Any]]:
    return context.user_data.get("last_op")

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
        if not (filtro.get(k) or "").strip():
            return k
    return None

async def _preguntar_slot(update: Update, slot: str, products: List[Dict[str,Any]]):
    if slot == "Categoria":
        await update.message.reply_text("¿Qué *categoría* es? (ej. 'pantalon', 'bermuda', 'camisa')", parse_mode="Markdown")
    elif slot == "Producto":
        await update.message.reply_text("¿Qué *modelo/producto* es? (ej. 'clasico', 'cargo', 'argon')", parse_mode="Markdown")
    elif slot == "Talle":
        await update.message.reply_text("¿Qué *talle*?", parse_mode="Markdown")
    elif slot == "Color":
        await update.message.reply_text("¿Qué *color*?", parse_mode="Markdown")
    else:
        await update.message.reply_text("Necesito un dato más…")

# ----------------- RESOLVER Y AJUSTAR (para 1 ítem) -----------------
async def _resolver_y_ajustar(update, context, accion, cantidad, precio_venta, filtro, products, idx, ws_prod, ws_movs):
    candidatos = _filtrar_por_campos(products, filtro)
    if not candidatos:
        candidatos = _buscar_textual(products, " ".join([v for v in filtro.values() if v]))

    if not candidatos:
        slot = _siguiente_slot_faltante(filtro)
        if slot:
            await _preguntar_slot(update, slot, products)
            _set_pendiente(context, {"accion":accion,"cantidad":cantidad,"precio_venta":precio_venta,"filtro":filtro,"t":datetime.now().timestamp()})
            return None
        await update.message.reply_text("No encontré coincidencias con esos datos. Indicá *categoría, producto, talle y color*.", parse_mode="Markdown")
        return None

    if len(candidatos) > 1:
        lista = "\n".join([f"- {p['Categoria']} {p['Producto']} {p['Talle']} {p['Color']} · código: {p.get('Codigo') or 's/cod'}" for p in candidatos[:12]])
        await update.message.reply_text(
            f"Encontré varias coincidencias:\n{lista}\n\nDecime el *código* exacto o especificá mejor (p. ej. 'pantalon cargo 42 verde').",
            parse_mode="Markdown"
        )
        _set_pendiente(context, {"accion":accion,"cantidad":cantidad,"precio_venta":precio_venta,"filtro":filtro,"t":datetime.now().timestamp()})
        return None

    # === ÚNICO PRODUCTO → AJUSTAR ===
    prod = candidatos[0]
    delta = abs(cantidad) if accion == "agregar" else -abs(cantidad)
    new_stock = max(0, prod["Stock"] + delta)
    _update_stock(ws_prod, prod, new_stock, idx)
    tipo = "entrada" if delta > 0 else "salida"
    precio_v = float(precio_venta or prod["Precio"] or 0)
    costo = float(prod["Costo"] or 0)
    _append_movement(ws_movs, tipo, prod, abs(delta), precio_v, costo)

    # guardar última operación para anular
    _set_last_operation(context, {
        "tipo": tipo, "prod_row": prod["row"], "old_stock": prod["Stock"], "new_stock": new_stock,
        "cantidad": abs(delta), "precio_venta": precio_v, "costo": costo,
        "prod_keys": {k: prod.get(k) for k in ["Codigo","Categoria","Producto","Talle","Color"]}
    })

    msg = f"✅ {('Sumé' if delta>0 else 'Desconté')} {abs(delta)} de {prod['Categoria']} {prod['Producto']} {prod['Talle']} {prod['Color']} (stock {prod['Stock']}→{new_stock})"
    if tipo == "salida":
        msg += f"\nGanancia estimada: {_money((precio_v - costo)*abs(delta))}"
    await update.message.reply_text(msg)
    return True

# ----------------- PARSE DE MÚLTIPLES ÍTEMS -----------------
def _split_items(text: str) -> List[str]:
    # corta por comas y conectores " y ", " e ", " + ", " / "
    # preserva la primera parte completa (que puede tener el verbo)
    t = text.strip()
    # Reemplazar conectores por comas para un split uniforme
    conectores = [r"\sy\s", r"\se\s", r"\s\+\s", r"\s/\s"]
    for c in conectores:
        t = re.sub(c, ", ", t, flags=re.IGNORECASE)
    partes = [p.strip() for p in t.split(",") if p.strip()]
    return partes

def _intencion_global(text_nrm: str) -> Optional[str]:
    return _detectar_intencion(text_nrm)

def _cantidad_precio_from_text(text: str) -> Tuple[int, Optional[float]]:
    t = _normalize(text)
    # Precio de venta (único)
    m = re.search(r"\$?\s*([\d\.]{1,3}(?:[\.\s]?\d{3})*(?:[\,\.]\d{1,2})?)", t)
    precio = None
    if m:
        try:
            val = m.group(1).replace(".", "").replace(" ", "").replace(",", ".")
            precio = float(val)
        except:
            pass
    # Cantidad por patrones maestros (si no, se define en _resolver_cantidad_y_talle→default 1)
    cant = _extraer_cantidad_desde_patrones(t)
    return (cant if cant else 1), precio

# ----------------- COMANDOS -----------------
async def _cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "👋 *Bot de Control de Stock*\n\n"
        "Ejemplos:\n"
        "• \"Vendí 2 pantalones cargo verde 44 y una camisa verde 42\"\n"
        "• \"Compré 10 guantes grip negros\"\n"
        "• \"Vendí 2 pantalones verde clasico 44, una camisa verde 42 y una camisa verde 44\"\n"
        "• Comandos: /reporte, /faltantes, /ganancias, /anular, /ping, /version, /estado\n"
        "Tip: podés indicar *código* o *SKU* para elegir exacto."
    )
    await update.message.reply_markdown(msg)

async def _cmd_reporte(update, context):
    _, ws, _ = _open_sheet()
    products, _ = _read_products(ws)
    if not products:
        await update.message.reply_text("No hay productos cargados.")
        return
    text = "📊 *Reporte de stock:*\n"
    for p in products[:200]:
        text += f"- {p['Categoria']} {p['Producto']} {p['Talle']} {p['Color']} | Stock: {p['Stock']} (mín: {p['Minimo']}) | Precio: {_money(p['Precio'])}\n"
    await update.message.reply_markdown(text[:4000])

async def _cmd_faltantes(update, context):
    _, ws, _ = _open_sheet()
    products, _ = _read_products(ws)
    falt = [p for p in products if p["Stock"] <= p["Minimo"]]
    if not falt:
        await update.message.reply_text("✅ No hay faltantes.")
        return
    text = "🚨 *Bajo stock:*\n"
    for p in falt[:200]:
        text += f"- {p['Categoria']} {p['Producto']} {p['Talle']} {p['Color']} ({p['Stock']} / mín {p['Minimo']})\n"
    await update.message.reply_markdown(text[:4000])

async def _cmd_ganancias(update, context):
    _, _, ws_movs = _open_sheet()
    total = _sum_ganancias(ws_movs)
    await update.message.reply_text(f"💰 Ganancia total: {_money(total)}")

async def _cmd_anular(update, context):
    sh, ws_prod, ws_movs = _open_sheet()
    products, idx = _read_products(ws_prod)
    last = _get_last_operation(context)
    if not last:
        await update.message.reply_text("No hay una operación reciente para anular.")
        return
    # recrear prod por row
    prod = None
    for p in products:
        if p["row"] == last["prod_row"]:
            prod = p
            break
    if not prod:
        await update.message.reply_text("No pude localizar el producto de la última operación.")
        return
    # deshacer
    delta = last["cantidad"] if last["tipo"] == "salida" else -last["cantidad"]
    new_stock = max(0, prod["Stock"] + delta)
    _update_stock(ws_prod, prod, new_stock, idx)
    _append_movement(ws_movs, "anulacion", prod, last["cantidad"], last["precio_venta"], last["costo"], nota="Deshacer última operación")
    _set_last_operation(context, None)
    await update.message.reply_text(
        f"↩️ Anulada la última operación: {prod['Categoria']} {prod['Producto']} {prod['Talle']} {prod['Color']} "
        f"(stock {prod['Stock']}→{new_stock})"
    )

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
            "openai": (pkg_resources.get_distribution("openai").version if OPENAI_API_KEY else "(deshabilitado)"),
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
                    ult.append(f"{r[0]} · {r[1]} · {r[3]} {r[4]} {r[5]} {r[6]} · cant {r[7]}")
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

# ----------------- MENSAJERÍA (texto libre) -----------------
async def _on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    text_nrm = _normalize(text)

    # Atajo para "anular" por texto
    if re.search(r"\b(anular|deshacer|deshace|undo)\b", text_nrm):
        await _cmd_anular(update, context)
        return

    sh, ws_prod, ws_movs = _open_sheet()
    products, idx = _read_products(ws_prod)

    # Si hay pendiente y el usuario responde un código, resolver por código directo
    pending = _estado_pendiente(context)
    if pending and _is_probable_code(text):
        prod = _find_by_code(text, products)
        if prod:
            accion = pending["accion"]
            cantidad = int(pending["cantidad"])
            precio_venta = pending.get("precio_venta")
            delta = abs(cantidad) if accion == "agregar" else -abs(cantidad)
            new_stock = max(0, prod["Stock"] + delta)
            _update_stock(ws_prod, prod, new_stock, idx)
            tipo = "entrada" if delta > 0 else "salida"
            precio_v = float(pr
e
cio_venta or prod["Precio"] or 0)
            costo = float(prod["Costo"] or 0)
            _append_movement(ws_movs, tipo, prod, abs(delta), precio_v, costo)
            _set_pendiente(context, None)
            _set_last_operation(context, {
                "tipo": tipo, "prod_row": prod["row"], "old_stock": prod["Stock"], "new_stock": new_stock,
                "cantidad": abs(delta), "precio_venta": precio_v, "costo": costo,
                "prod_keys": {k: prod.get(k) for k in ["Codigo","Categoria","Producto","Talle","Color"]}
            })
            msg = f"✅ {('Sumé' if delta>0 else 'Desconté')} {abs(delta)} de {prod['Categoria']} {prod['Producto']} {prod['Talle']} {prod['Color']} (stock {prod['Stock']}→{new_stock})"
            if tipo == "salida":
                msg += f"\nGanancia estimada: {_money((precio_v - costo)*abs(delta))}"
            await update.message.reply_text(msg)
            return

    # Parse general (IA opcional)
    parsed = _nlp_parse(text)
    intent = parsed.get("intent") or "ajustar_stock"

    if intent in ("reporte","faltantes","ganancias"):
        if intent == "reporte":
            await _cmd_reporte(update, context);  return
        if intent == "faltantes":
            await _cmd_faltantes(update, context);  return
        if intent == "ganancias":
            await _cmd_ganancias(update, context);  return

    if intent == "anular":
        await _cmd_anular(update, context);  return

    # Multi-ítems: dividir en partes
    partes = _split_items(text)
    accion_global = _intencion_global(text_nrm)
    _, _, ws_movs = _open_sheet()

    # Procesar cada parte
    hubo_accion = False
    for i, parte in enumerate(partes):
        # cantidad/precio por parte
        cant_hint, precio_hint = _cantidad_precio_from_text(parte)
        # entidades por parte
        filtro = _extract_entities_from_query(products, parte)
        # fijar acción
        accion = accion_global or _detectar_intencion(_normalize(parte)) or "descontar"  # si no se dice, asumimos venta
        # cantidad final (preferimos el hint de la parte; si IA devolvió items, podríamos usarlo, pero mantenemos robusto)
        cantidad = filtro.get("_CantidadHint") or cant_hint or 1
        precio_venta = precio_hint  # puede ser None, se usará precio de lista si falta

        ok = await _resolver_y_ajustar(update, context, accion, int(cantidad), precio_venta, filtro, products, idx, ws_prod, ws_movs)
        if ok:
            hubo_accion = True
        else:
            # si quedó pendiente de slot-filling, no seguir forzando las otras partes aún
            pending = _estado_pendiente(context)
            if pending:
                break

    if not hubo_accion and not _estado_pendiente(context):
        await update.message.reply_text("No pude interpretar la orden. Indicá *categoría, producto, talle y color*. Ej: 'vendí 2 pantalones cargo verde 44'", parse_mode="Markdown")

# ----------------- MAIN -----------------
def main():
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", _cmd_start))
    app.add_handler(CommandHandler("reporte", _cmd_reporte))
    app.add_handler(CommandHandler("faltantes", _cmd_faltantes))
    app.add_handler(CommandHandler("ganancias", _cmd_ganancias))
    app.add_handler(CommandHandler("anular", _cmd_anular))
    app.add_handler(CommandHandler("ping", _cmd_ping))
    app.add_handler(CommandHandler("version", _cmd_version))
    app.add_handler(CommandHandler("estado", _cmd_estado))

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_text))

    log.info("Bot corriendo…")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
