# 🤖 Control de Stock con IA — Telegram Bot

Un bot de Telegram que administra stock en tiempo real usando Google Sheets, con IA (OpenAI) para entender lenguaje natural.

## 🚀 Funciones principales
- Suma o descuenta productos del stock
- Usa columnas separadas para producto, talle y color
- Calcula ganancias y genera reportes
- Se conecta en línea con Google Sheets
- Funciona 24 hs alojado en Render

## 🧱 Tecnologías
Python 3.12 — python-telegram-bot — gspread — Google Sheets API — OpenAI API — Render — GitHub

## ⚙️ Instalación local
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt

## 📊 Google Sheets
**Productos**  
`Codigo | Descripcion | Producto | Talle | Color | Costo | Precio | Stock | Minimo | SKU`  

**Movimientos**  
`Fecha | Tipo | Codigo | Producto | Talle | Color | Descripcion | Cantidad | PrecioVenta | CostoUnitario | Ganancia`

## 🧠 Ejemplo de uso
- “sumá 3 pantalones talle M azul”
- “vendí 2 botines negros 42 a $34000”
- “reporte”
- “faltantes”

## ☁️ Render
Build Command: `pip install -r requirements.txt`  
Start Command: `python bot.py`

Variables de entorno:  
`TELEGRAM_TOKEN`, `OPENAI_API_KEY`, `OPENAI_MODEL`, `GOOGLE_SHEET_ID`, `GOOGLE_SERVICE_ACCOUNT_JSON`
