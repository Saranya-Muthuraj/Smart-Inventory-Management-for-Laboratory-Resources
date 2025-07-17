import streamlit as st
import pandas as pd
import speech_recognition as sr
from datetime import datetime, timedelta
from prophet import Prophet
import re
import difflib
import plotly.graph_objs as go

# Page configuration
st.set_page_config(page_title="Smart Inventory", layout="centered")
st.markdown("<h1 style='text-align: center;'>🧪 Smart Lab Inventory System</h1>", unsafe_allow_html=True)

# Load inventory data
inv_df = pd.read_csv("lab_inventory.csv")
inv_df["ExpiryDate"] = pd.to_datetime(inv_df["ExpiryDate"])

VALID_RESOURCES = inv_df["Resource"].str.lower().tolist()

# ---------- FUNCTIONS ----------

def correct_resource_name(word):
    matches = difflib.get_close_matches(word, VALID_RESOURCES, n=1, cutoff=0.7)
    return matches[0] if matches else word

def parse_command(command):
    command = command.lower()
    if "use" in command or "return" in command:
        action = "use" if "use" in command else "return"
        pattern = rf"{action}\s+(\d+)\s*(units|ml|g|pcs|pairs)?\s*(of)?\s*(.+)"
        match = re.match(pattern, command)
        if match:
            qty = int(match.group(1))
            item = correct_resource_name(match.group(4).strip().lower())
            return action, qty, item
    return None, None, None

def get_voice_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening...")
        audio = r.listen(source)
    try:
        return r.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, could not understand the audio"
    except sr.RequestError:
        return "Sorry, request to Google Speech Recognition failed"

def handle_voice_command(command):
    action, qty, item = parse_command(command)
    if qty is None or item is None:
        return "Invalid command format."

    for idx, row in inv_df.iterrows():
        if row["Resource"].lower() == item:
            if action == "use":
                if inv_df.loc[idx, "TotalQty"] >= qty:
                    inv_df.loc[idx, "TotalQty"] -= qty
                    log_usage(action, item, qty)
                    inv_df.to_csv("lab_inventory.csv", index=False)
                    return f"{qty} units of {item} used from inventory."
                else:
                    return f"Not enough {item} in inventory."
            elif action == "return":
                inv_df.loc[idx, "TotalQty"] += qty
                log_usage(action, item, qty)
                inv_df.to_csv("lab_inventory.csv", index=False)
                return f"{qty} units of {item} returned to inventory."
    return f"{item} not found in inventory."

def log_usage(action, item, qty):
    log_df = pd.DataFrame([[datetime.now().strftime("%Y-%m-%d %H:%M:%S"), action, item, qty]],
                          columns=["Date", "Action", "Resource", "Qty"])
    try:
        existing = pd.read_csv("lab_usage.csv")
        combined = pd.concat([existing, log_df], ignore_index=True)
    except FileNotFoundError:
        combined = log_df
    combined.to_csv("lab_usage.csv", index=False)

# ---------- UI ----------

with st.expander("💬 Show Inventory Data"):
    st.dataframe(inv_df.style.format({"ExpiryDate": lambda x: x.strftime("%Y-%m-%d")}), use_container_width=True)

st.subheader("🧑‍⚕ Voice Inventory Assistant")
st.markdown("### 🎙 Speak your command")

if st.button("🎤 Start Voice Command"):
    voice_cmd = get_voice_command()
    st.write("Your command (auto-filled from mic)")
    st.code(voice_cmd)
    result = handle_voice_command(voice_cmd)
    st.success(result)
    st.rerun()

st.markdown("---")
manual = st.text_input("Or type your command")
if manual:
    result = handle_voice_command(manual)
    st.success(result)
    st.rerun()

# ---------- Forecasting with Prophet ----------
st.markdown("---")
st.subheader("📈 Demand Forecasting for All Resources")

try:
    logs_df = pd.read_csv("lab_usage.csv")
    logs_df["Date"] = pd.to_datetime(logs_df["Date"], format='mixed', errors='coerce')
    logs_df.dropna(subset=["Date"], inplace=True)
    usage_logs = logs_df[logs_df["Action"].str.lower() == "use"]

    if usage_logs.empty:
        st.warning("No 'use' records found in lab_usage.csv.")
    else:
        forecast_days = st.slider("📅 Days to Forecast", min_value=7, max_value=30, value=14)

        for resource_name in usage_logs["Resource"].unique():
            df_res = usage_logs[usage_logs["Resource"] == resource_name][["Date", "Qty"]].rename(columns={"Date": "ds", "Qty": "y"})

            if df_res.shape[0] < 2:
                st.info(f"⏳ Not enough data to forecast '{resource_name}'")
                continue

            st.markdown(f"### 🔮 Forecast for **{resource_name}**")

            model = Prophet()
            model.fit(df_res)
            future = model.make_future_dataframe(periods=forecast_days)
            forecast = model.predict(future)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_res["ds"], y=df_res["y"], name="Actual Usage", line=dict(color="blue")))
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast", line=dict(color="orange")))
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], name="Upper Bound", line=dict(color="lightgrey"), showlegend=False))
            fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], name="Lower Bound", line=dict(color="lightgrey"), fill='tonexty', showlegend=False))

            fig.update_layout(title=f"{resource_name} Usage Forecast", xaxis_title="Date", yaxis_title="Used Quantity", height=450)
            st.plotly_chart(fig, use_container_width=True)

            forecast_display = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(forecast_days)
            forecast_display.columns = ["Date", "Predicted Usage", "Lower Estimate", "Upper Estimate"]
            st.dataframe(forecast_display, use_container_width=True)
except Exception as e:
    st.error(f"Error while forecasting: {e}")

# ---------- Logs ----------
st.markdown("---")
st.subheader("📋 Usage Logs")
try:
    logs_df = pd.read_csv("lab_usage.csv")
    st.dataframe(logs_df.tail(10), use_container_width=True)
except FileNotFoundError:
    st.info("No usage logs found yet.")

# ---------- Restocking Alerts ----------
st.markdown("---")
st.subheader("🛎️ Automated Restocking Alerts")

restock_threshold = 10
low_stock = inv_df[inv_df["TotalQty"] < restock_threshold]

if low_stock.empty:
    st.success("✅ All inventory levels are sufficient.")
else:
    st.warning("⚠️ The following items need restocking:")
    st.dataframe(low_stock[["Resource", "TotalQty"]], use_container_width=True)

# ---------- Expiry Monitoring ----------
st.markdown("---")
st.subheader("⏳ Expiry Monitoring")

days_to_check = 30
today = pd.Timestamp.today()
expiring_soon = inv_df[
    (inv_df["ExpiryDate"] > today) & 
    (inv_df["ExpiryDate"] <= today + pd.Timedelta(days=days_to_check))
]

if expiring_soon.empty:
    st.success("✅ No items are expiring in the next 30 days.")
else:
    st.warning(f"⚠️ The following items are expiring within {days_to_check} days:")
    st.dataframe(expiring_soon[["Resource", "ExpiryDate"]].sort_values("ExpiryDate"), use_container_width=True)

# ---------- Footer ----------
st.markdown("---")
st.caption("Built for Smart Lab Inventory Management 🧠")