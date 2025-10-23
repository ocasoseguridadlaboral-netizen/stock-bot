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

PRODUCTS_HEADERS = ["Codigo", "Descripcion", "Producto", "Talle", "Color", "Costo", "Precio", "Stock", "Minimo", "SKU"]
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
    "gorras":"gorra", "camperas":"campera", "buzos":"buzo",
}
def _singularize(word: str) -> str:
    w = _normalize(word)
    if w in _SINGULAR_MAP: return _SINGULAR_MAP[w]
    # reglas simples
    if w.endswith("es") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and len(w) > 3:
        return w[:-1]
    return w

def _money(x: float) -> str:
    return f"${x:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")

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
        def get_text(col): return str(row[idx[col]]).strip() if col in idx and idx[col] < len(row) else ""
        def get_num(col):
            try: return float(str(row[idx[col]]).replace(",", ".") or 0)
            except: return 0
        p = {
            "row": r_i,
            "Codigo": get_text("codigo"),
            "Descripcion": get_text("descripcion"),
            "Producto": get_text("producto"),
            "Talle": get_text("talle"),
            "Color": get_text("color"),
            "Costo": get_num("costo"),
            "Precio": get_num("precio"),
            "Stock": int(get_num("stock")),
            "Minimo": int(get_num("minimo")),
            "SKU": get_text("sku")
        }
        if p["Codigo"] or p["Descripcion"] or p["Producto"]:
            products.append(p)
    return products, idx

# ----------------- EXTRACCIÓN DE ENTIDADES DESDE EL MENSAJE -----------------
VERBOS_SUMA = {"compre","compré","compra","agrega","agregá","agregar","sumar","sumá","suma","entrada","entraron","ingresa","ingresá","agrego","ingreso"}
VERBOS_RESTA = {"vendi","vendí","venta","vendimos","desconta","descontá","descontar","restar","resta","salida","salieron","retiro","retiré","descuento"}

def _detectar_intencion(text_nrm: str) -> Optional[str]:
    tokens = text_nrm.split()
    if any(v in tokens for v in VERBOS_RESTA): return "descontar"
    if any(v in tokens for v in VERBOS_SUMA): return "agregar"
    # heurística: si aparece "vend..." en el texto
    if re.search(r"\bvend", text_nrm): return "descontar"
    if re.search(r"\bcompr", text_nrm): return "agregar"
    return None

def _extraer_cantidad(text_nrm: str) -> int:
    m = re.search(r"\b(\d{1,4})\b", text_nrm)
    if m:
        try: return int(m.group(1))
        except: pass
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

def _catalogo_valores(products: List[Dict[str,Any]]) -> Tuple[List[str], List[str], List[str]]:
    prods = sorted({ _normalize(p.get("Producto","")) for p in products if p.get("Producto") }, key=len, reverse=True)
    colores = sorted({ _normalize(p.get("Color","")) for p in products if p.get("Color") }, key=len, reverse=True)
    talles = sorted({ _normalize(p.get("Talle","")) for p in products if p.get("Talle") }, key=len, reverse=True)
    return prods, colores, talles

def _pick_one_from(text_nrm: str, candidates: List[str]) -> Optional[str]:
    # match exact substring de candidato más largo primero
    for c in candidates:
        if not c: 
            continue
        if c in text_nrm:
            return c
    return None

def _extract_entities_from_query(products: List[Dict[str,Any]], text: str) -> Dict[str, Optional[str]]:
    t = _normalize(text)
    # singularizar cada token para mejorar match (“pantalones” => “pantalon”)
    tokens = [_singularize(tok) for tok in t.split()]
    t_sing = " ".join(tokens)

    prods_cat, colores_cat, talles_cat = _catalogo_valores(products)

    producto = _pick_one_from(t_sing, prods_cat)
    color = _pick_one_from(t_sing, colores_cat)

    # talle: primero “talle X” o “t X”
    m = re.search(r"\b(?:talle|t)\s*([a-z0-9\-]+)\b", t_sing)
    if m:
        talle = m.group(1)
    else:
        # si en catálogo hay talles numéricos/letras, buscar token exacto
        talle = None
        for tok in tokens:
            if tok in talles_cat:
                talle = tok; break
            # números como 44, 42, 36
            if re.fullmatch(r"\d{2}", tok) and tok in talles_cat:
                talle = tok; break

    return {"Producto": producto, "Color": color, "Talle": talle}

# ----------------- BÚSQUEDA Y MATCHING -----------------
def _filtrar_por_campos(products: List[Dict[str,Any]], f: Dict[str, Optional[str]]) -> List[Dict[str,Any]]:
    # filtra por campos especificados (normalizados)
    def ok(p, k):
        v = _normalize(p.get(k,""))
        q = _normalize(f.get(k) or "")
        return (not q) or (q and q == v)

    res = [p for p in products if ok(p,"Producto") and ok(p,"Color") and ok(p,"Talle")]
    return res

def _buscar_textual(products: List[Dict[str,Any]], query: str) -> List[Dict[str,Any]]:
    q = _normalize(query)
    toks = set(q.split())
    out = []
    for p in products:
        texto = _normalize(" ".join([
            p.get("Codigo",""), p.get("Descripcion",""), p.get("Producto",""),
            p.get("Talle",""), p.get("Color",""), p.get("SKU","")
        ]))
        if q in texto or toks.issubset(set(texto.split())):
            out.append(p)
    return out

# ----------------- IA (opcional) -----------------
def _nlp_parse(text: str) -> Dict[str, Any]:
    """
    Intentamos OpenAI; si falla o no hay key, usamos reglas locales robustas.
    """
    # 1) Reglas locales mejoradas
    t = _normalize(text)
    accion = _detectar_intencion(t)  # "agregar" | "descontar" | None
    cantidad = _extraer_cantidad(t)
    precio = _extraer_precio(t)  # puede ser None

    base = {"intent":"ajustar_stock","data":{"accion":accion,"cantidad":cantidad,"query":text,"precio_venta":precio}}

    if OPENAI_API_KEY and OpenAI:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            system = (
                "Sos un parser de español para control de stock. "
                "Devolvé JSON con 'intent' y 'data'. "
                "intents: 'ajustar_stock','reporte','faltantes','ganancias'. "
                "Para 'ajustar_stock' incluí: {'accion':'agregar'|'descontar','cantidad':int,'query':str,'precio_venta':float|null}."
            )
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[{"role":"system","content":system},{"role":"user","content":text}],
                response_format={"type":"json_object"},
                temperature=0
            )
            data = json.loads(resp.choices[0].message.content)
            # combinamos: si OpenAI no detectó bien la acción o cantidad, usamos nuestra heurística
            d = data.get("data",{})
            if not data.get("intent"): data["intent"] = "ajustar_stock"
            if not d.get("accion"): d["accion"] = accion
            if not d.get("cantidad"): d["cantidad"] = cantidad
            if "precio_venta" not in d: d["precio_venta"] = precio
            data["data"] = d
            return data
        except Exception as e:
            log.warning(f"OpenAI deshabilitado por error: {e}")

    return base

# ----------------- MOVIMIENTOS / STOCK -----------------
def _update_stock(ws, prod, new_stock, idx):
    col = (idx.get("stock") or 7) + 1
    ws.update_cell(prod["row"], col, new_stock)

def _append_movement(ws_movs, tipo, prod, cantidad, precio_venta, costo_unit):
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ganancia = (precio_venta - costo_unit) * abs(int(cantidad)) if tipo == "salida" else 0
    ws_movs.append_row([
        fecha, tipo, prod.get("Codigo",""), prod.get("Producto",""),
        prod.get("Talle",""), prod.get("Color",""), prod.get("Descripcion",""),
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

# ----------------- HANDLERS BÁSICOS -----------------
async def _cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "👋 *Bot de Control de Stock*\n\n"
        "Ejemplos:\n"
        "• 'Sumá 3 pantalones verde 44'\n"
        "• 'Vendí 2 botines negros 42 a $34000'\n"
        "• 'Compré 5 pantalones verde clasico 44'\n"
        "• 'Mostrame los faltantes' | 'Ganancias'\n\n"
        "Comandos: /reporte, /faltantes, /ganancias, /ping, /version, /estado"
    )
    await update.message.reply_markdown(msg)

async def _cmd_reporte(update, context):
    _, ws, _ = _open_sheet()
    products, _ = _read_products(ws)
    if not products:
        await update.message.reply_text("No hay productos cargados.")
        return
    text = "📊 *Reporte de stock:*\n"
    for p in products:
        text += f"- {p['Producto']} {p['Talle']} {p['Color']} | Stock: {p['Stock']} (mín: {p['Minimo']}) | Precio: {_money(p['Precio'])}\n"
    await update.message.reply_markdown(text[:4000])

async def _cmd_faltantes(update, context):
    _, ws, _ = _open_sheet()
    products, _ = _read_products(ws)
    falt = [p for p in products if p["Stock"] <= p["Minimo"]]
    if not falt:
        await update.message.reply_text("✅ No hay faltantes.")
        return
    text = "🚨 *Bajo stock:*\n"
    for p in falt:
        text += f"- {p['Producto']} {p['Talle']} {p['Color']} ({p['Stock']} / mín {p['Minimo']})\n"
    await update.message.reply_markdown(text[:4000])

async def _cmd_ganancias(update, context):
    _, _, ws_movs = _open_sheet()
    total = _sum_ganancias(ws_movs)
    await update.message.reply_text(f"💰 Ganancia total: {_money(total)}")

# ----------------- SLOT-FILLING Y AJUSTE -----------------
def _estado_pendiente(context: ContextTypes.DEFAULT_TYPE) -> Optional[Dict[str,Any]]:
    return context.user_data.get("pending_adjust")

def _set_pendiente(context: ContextTypes.DEFAULT_TYPE, data: Optional[Dict[str,Any]]):
    if data is None:
        context.user_data.pop("pending_adjust", None)
    else:
        context.user_data["pending_adjust"] = data

def _siguiente_slot_faltante(filtro: Dict[str, Optional[str]]) -> Optional[str]:
    for k in ["Producto","Talle","Color"]:
        if not filtro.get(k):
            return k
    return None

async def _preguntar_slot(update: Update, slot: str, products: List[Dict[str,Any]]):
    if slot == "Producto":
        opciones = sorted({p["Producto"] for p in products if p.get("Producto")})
        extra = f"\nEjemplos: {', '.join(opciones[:5])}" if opciones else ""
        await update.message.reply_text(f"¿Qué *producto* es? (ej. 'pantalon clasico'){extra}", parse_mode="Markdown")
    elif slot == "Talle":
        opciones = sorted({p["Talle"] for p in products if p.get("Talle")})
        extra = f"\nTalles: {', '.join(opciones[:10])}" if opciones else ""
        await update.message.reply_text(f"¿Qué *talle*?", parse_mode="Markdown")
    elif slot == "Color":
        opciones = sorted({p["Color"] for p in products if p.get("Color")})
        extra = f"\nColores: {', '.join(opciones[:10])}" if opciones else ""
        await update.message.reply_text(f"¿Qué *color*?", parse_mode="Markdown")
    else:
        await update.message.reply_text("Necesito un dato más…")

async def _resolver_y_ajustar(update, context, accion, cantidad, precio_venta, filtro, products, idx, ws_prod, ws_movs):
    # filtrar por lo que ya tengo
    candidatos = _filtrar_por_campos(products, filtro)
    if not candidatos:
        # último intento: búsqueda textual completa
        candidatos = _buscar_textual(products, " ".join([v for v in filtro.values() if v]))

    if not candidatos:
        slot = _siguiente_slot_faltante(filtro)
        if slot:
            await _preguntar_slot(update, slot, products)
            _set_pendiente(context, {"accion":accion,"cantidad":cantidad,"precio_venta":precio_venta,"filtro":filtro})
            return
        await update.message.reply_text("No encontré coincidencias con esos datos. Probá indicar *producto, talle y color*.", parse_mode="Markdown")
        return

    if len(candidatos) > 1:
        lista = "\n".join([f"- {p['Producto']} {p['Talle']} {p['Color']} [código: {p.get('Codigo') or 's/cod'}]" for p in candidatos[:10]])
        await update.message.reply_text(f"Encontré varias coincidencias:\n{lista}\n\nDecime el *código* exacto o especificá mejor (p. ej. producto y color).", parse_mode="Markdown")
        _set_pendiente(context, {"accion":accion,"cantidad":cantidad,"precio_venta":precio_venta,"filtro":filtro})
        return

    # 1 solo producto → ajustar
    prod = candidatos[0]
    delta = abs(cantidad) if accion == "agregar" else -abs(cantidad)
    new_stock = max(0, prod["Stock"] + delta)
    _update_stock(ws_prod, prod, new_stock, idx)
    tipo = "entrada" if delta > 0 else "salida"
    precio_v = float(precio_venta or prod["Precio"] or 0)
    costo = float(prod["Costo"] or 0)
    _append_movement(ws_movs, tipo, prod, abs(delta), precio_v, costo)

    msg = f"✅ {('Sumé' if delta>0 else 'Desconté')} {abs(delta)} de {prod['Producto']} {prod['Talle']} {prod['Color']} (stock {prod['Stock']}→{new_stock})"
    if tipo == "salida":
        msg += f"\nGanancia estimada: {_money((precio_v - costo)*abs(delta))}"
    await update.message.reply_text(msg)

async def _on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    sh, ws_prod, ws_movs = _open_sheet()
    products, idx = _read_products(ws_prod)

    # si estoy completando slots
    pending = _estado_pendiente(context)
    if pending:
        # actualizar el filtro con lo que respondió el usuario
        t = _normalize(text)
        slot = _siguiente_slot_faltante(pending["filtro"])
        if slot:
            # intentar mapear respuesta al catálogo
            if slot == "Producto":
                # permitir singular plural
                pending["filtro"]["Producto"] = _pick_one_from(" "+_singularize(t)+" ", [_normalize(p["Producto"]) for p in products if p.get("Producto")])
            elif slot == "Talle":
                pending["filtro"]["Talle"] = t.strip()
            elif slot == "Color":
                pending["filtro"]["Color"] = _pick_one_from(" "+t+" ", [_normalize(p["Color"]) for p in products if p.get("Color")]) or t.strip()
        # reintentar resolver
        await _resolver_y_ajustar(update, context, pending["accion"], int(pending["cantidad"]), pending.get("precio_venta"), pending["filtro"], products, idx, ws_prod, ws_movs)
        return

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

    # ajustar_stock
    accion = data.get("accion")
    cantidad = int(data.get("cantidad") or 1)
    precio_venta = data.get("precio_venta")

    # extraer entidades desde el texto con catálogo real
    filtro = _extract_entities_from_query(products, text)

    # si falta acción, decidir por heurística (por defecto: venta = descontar)
    if not accion:
        accion = "descontar"  # elección conservadora si dice “vendí …”
        if re.search(r"\bcompr|agreg|sum", _normalize(text)): accion = "agregar"

    # si faltan slots, pregunto de a uno antes de fallar
    await _resolver_y_ajustar(update, context, accion, cantidad, precio_venta, filtro, products, idx, ws_prod, ws_movs)

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
    # Sheets
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
    # OpenAI
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
        raise RuntimeError("Faltan variables de entorno.")
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # Comandos
    app.add_handler(CommandHandler("start", _cmd_start))
    app.add_handler(CommandHandler("reporte", _cmd_reporte))
    app.add_handler(CommandHandler("faltantes", _cmd_faltantes))
    app.add_handler(CommandHandler("ganancias", _cmd_ganancias))
    app.add_handler(CommandHandler("ping", _cmd_ping))
    app.add_handler(CommandHandler("version", _cmd_version))
    app.add_handler(CommandHandler("estado", _cmd_estado))

    # Mensajes libres y aclaraciones (slot-filling)
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_text))

    log.info("Bot corriendo…")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
