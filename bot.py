import os, json, re, logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import gspread
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters

# ==== IA (OpenAI) ====
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
        if p["Codigo"] or p["Descripcion"]:
            products.append(p)
    return products, idx

# ----------------- IA -----------------
def _nlp_parse(text: str) -> Dict[str, Any]:
    if OPENAI_API_KEY and OpenAI:
        client = OpenAI(api_key=OPENAI_API_KEY)
        system = (
            "Sos un asistente para un sistema de control de stock. "
            "Analizá el mensaje y devolvé JSON con 'intent' y 'data'. "
            "intents: 'ajustar_stock', 'reporte', 'faltantes', 'ganancias', 'agregar_producto'. "
            "Para 'ajustar_stock': {'accion':'agregar'|'descontar','cantidad':int,'query':str,'precio_venta':float opcional}."
        )
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": text}],
            response_format={"type": "json_object"},
            temperature=0
        )
        try:
            data = json.loads(resp.choices[0].message.content)
            return data
        except:
            pass
    # fallback simple
    t = text.lower()
    if "faltante" in t: return {"intent": "faltantes", "data": {}}
    if "reporte" in t: return {"intent": "reporte", "data": {}}
    if "ganancia" in t: return {"intent": "ganancias", "data": {}}
    m_add = re.search(r"(agrega|sumar|entrad[ao])\s+(\d+)", t)
    m_sub = re.search(r"(desconta|vend[íi]|restar|salid[ao])\s+(\d+)", t)
    if m_add:
        return {"intent": "ajustar_stock", "data": {"accion": "agregar", "cantidad": int(m_add.group(2)), "query": t}}
    if m_sub:
        return {"intent": "ajustar_stock", "data": {"accion": "descontar", "cantidad": int(m_sub.group(2)), "query": t}}
    return {"intent": "reporte", "data": {}}

# ----------------- UTIL -----------------
def _find_product(query: str, products: List[Dict[str, Any]]):
    q = query.strip().lower()
    matches = []
    for p in products:
        texto = f"{p.get('Codigo','')} {p.get('Descripcion','')} {p.get('Producto','')} {p.get('Talle','')} {p.get('Color','')} {p.get('SKU','')}".lower()
        if q in texto:
            matches.append(p)
    if not matches:
        palabras = q.split()
        for p in products:
            texto = f"{p.get('Descripcion','')} {p.get('Producto','')} {p.get('Talle','')} {p.get('Color','')}".lower()
            if all(w in texto for w in palabras):
                matches.append(p)
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        return matches
    return None

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

def _money(x): return f"${x:,.2f}".replace(",", "X").replace(".", ",").replace("X",".")

# ----------------- HANDLERS -----------------
async def _cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "👋 *Bienvenido al bot de Control de Stock*\n\n"
        "Podés decir cosas como:\n"
        "• 'Sumá 3 pantalones talle M azul'\n"
        "• 'Vendí 2 botines negros 42 a $34000'\n"
        "• 'Mostrame los faltantes'\n"
        "• 'Ganancias'\n\n"
        "Comandos:\n"
        "/reporte, /faltantes, /ganancias, /agregar, /descontar"
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
    text = "🚨 *Productos con bajo stock:*\n"
    for p in falt:
        text += f"- {p['Producto']} {p['Talle']} {p['Color']} ({p['Stock']} / mín {p['Minimo']})\n"
    await update.message.reply_markdown(text[:4000])

async def _cmd_ganancias(update, context):
    _, _, ws_movs = _open_sheet()
    total = _sum_ganancias(ws_movs)
    await update.message.reply_text(f"💰 Ganancia total: {_money(total)}")

# ----------------- AJUSTES DE STOCK -----------------
async def _ajustar_stock(update, accion, query, cantidad, precio_venta):
    sh, ws_prod, ws_movs = _open_sheet()
    products, idx = _read_products(ws_prod)
    prod = _find_product(query, products)
    if not prod:
        await update.message.reply_text(f"No encontré coincidencias para: \"{query}\"")
        return
    if isinstance(prod, list):
        opciones = "\n".join([
            f"- {p['Producto']} {p['Talle']} {p['Color']} [Código: {p['Codigo'] or 's/cod'}]" for p in prod
        ])
        await update.message.reply_text(
            f"Encontré varios productos que coinciden con \"{query}\":\n{opciones}\n\nEscribí el *código* exacto."
        )
        context.user_data["pending_action"] = {"accion": accion, "cantidad": cantidad, "precio_venta": precio_venta}
        return
    delta = abs(cantidad) if accion == "agregar" else -abs(cantidad)
    new_stock = max(0, prod["Stock"] + delta)
    _update_stock(ws_prod, prod, new_stock, idx)
    tipo = "entrada" if delta > 0 else "salida"
    precio_v = float(precio_venta or prod["Precio"] or 0)
    costo = float(prod["Costo"] or 0)
    _append_movement(ws_movs, tipo, prod, delta, precio_v, costo)
    msg = f"✅ {('Sumé' if delta>0 else 'Desconté')} {abs(delta)} de {prod['Producto']} {prod['Talle']} {prod['Color']} (stock {prod['Stock']}→{new_stock})"
    if tipo == "salida":
        msg += f"\nGanancia estimada: {_money((precio_v - costo)*abs(delta))}"
    await update.message.reply_text(msg)

async def _on_clarification(update, context):
    text = update.message.text.strip()
    pending = context.user_data.pop("pending_action", None)
    if not pending:
        await update.message.reply_text("No hay acción pendiente.")
        return
    await _ajustar_stock(update, pending["accion"], text, pending["cantidad"], pending.get("precio_venta"))

async def _on_text(update, context):
    text = update.message.text
    parsed = _nlp_parse(text)
    intent = parsed.get("intent")
    data = parsed.get("data", {})
    if intent == "ajustar_stock":
        await _ajustar_stock(update, data.get("accion"), data.get("query"), int(data.get("cantidad",0)), data.get("precio_venta"))
    elif intent == "reporte":
        await _cmd_reporte(update, context)
    elif intent == "faltantes":
        await _cmd_faltantes(update, context)
    elif intent == "ganancias":
        await _cmd_ganancias(update, context)
    else:
        await _cmd_reporte(update, context)

# ----------------- MAIN -----------------
def main():
    if not TELEGRAM_TOKEN or not GOOGLE_SHEET_ID:
        raise RuntimeError("Faltan variables de entorno.")
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", _cmd_start))
    app.add_handler(CommandHandler("reporte", _cmd_reporte))
    app.add_handler(CommandHandler("faltantes", _cmd_faltantes))
    app.add_handler(CommandHandler("ganancias", _cmd_ganancias))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _on_text))
    app.add_handler(MessageHandler(filters.TEXT & filters.REPLY, _on_clarification))
    log.info("Bot corriendo…")
    app.run_polling(drop_pending_updates=True)

if __name__ == "__main__":
    main()
