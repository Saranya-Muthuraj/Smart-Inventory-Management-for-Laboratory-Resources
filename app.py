import streamlit as st
import pandas as pd
import datetime
from prophet import Prophet
import plotly.graph_objs as go
from streamlit_extras.add_vertical_space import add_vertical_space

st.set_page_config(page_title="🧪 Smart Lab Inventory", layout="wide")
st.title("🧪 Smart Lab Inventory System")

# Load lab usage data
@st.cache_data
def load_data():
    return pd.read_csv("lab_usage.csv", parse_dates=["Date"])

df = load_data()
resources = df["Resource"].unique()

# === Voice Assistant Section ===
st.header("🗣️ Voice Inventory Assistant")

with st.expander("📄 Show Inventory Data"):
    st.dataframe(df.tail(10))

# JavaScript to access browser speech recognition
st.markdown("### 🎤 Speak your command")
st.markdown("""
<script>
  var recognition = new webkitSpeechRecognition();
  recognition.continuous = false;
  recognition.lang = 'en-US';

  function startDictation() {
    recognition.start();
    recognition.onresult = function(event) {
      var transcript = event.results[0][0].transcript;
      document.getElementById("output").value = transcript;
      document.getElementById("output").dispatchEvent(new Event("change"));
    };
  }
</script>

<button onclick="startDictation()">🎙️ Start Voice Command</button>
<br><br>
<input type="text" id="output" placeholder="Your command will appear here..." style="width: 100%; padding: 10px;">
""", unsafe_allow_html=True)

user_input = st.text_input("Your Command (auto-filled from mic)", key="voice_input")

def handle_command(cmd):
    cmd = cmd.lower()
    if "what items" in cmd or "list items" in cmd:
        return "You have: " + ", ".join(resources)
    elif "do we have" in cmd:
        for item in resources:
            if item.lower() in cmd:
                return f"Yes, {item} is available."
        return "Item not found."
    elif "use" in cmd:
        try:
            parts = cmd.split("use")[1].strip().split(" units of ")
            qty = int(parts[0])
            item = parts[1].title()
            df_item = df[df["Resource"].str.lower() == item.lower()]
            if df_item.empty:
                return f"{item} not found."
            last_idx = df_item.index[-1]
            df.loc[last_idx, "UsedQty"] += qty
            df.to_csv("lab_usage.csv", index=False)
            return f"{qty} units of {item} used and updated."
        except:
            return "Could not process the usage command."
    elif "return" in cmd:
        try:
            parts = cmd.split("return")[1].strip().split(" units of ")
            qty = int(parts[0])
            item = parts[1].title()
            df_item = df[df["Resource"].str.lower() == item.lower()]
            if df_item.empty:
                return f"{item} not found."
            last_idx = df_item.index[-1]
            df.loc[last_idx, "UsedQty"] = max(0, df.loc[last_idx, "UsedQty"] - qty)
            df.to_csv("lab_usage.csv", index=False)
            return f"{qty} units of {item} returned and updated."
        except:
            return "Could not process the return command."
    else:
        return "I didn’t understand. Try saying: 'Do we have Gloves?', 'Use 10 units of Ethanol', etc."

if user_input:
    response = handle_command(user_input)
    st.success(f"🗣️ {response}")
    add_vertical_space(1)

# === Demand Forecasting Section ===
st.header("📈 Demand Forecasting")

resource = st.selectbox("Select a Resource to Forecast", resources)
df_res = df[df["Resource"] == resource][["Date", "UsedQty"]].rename(columns={"Date": "ds", "UsedQty": "y"})
periods = st.slider("📅 Days to Forecast", min_value=7, max_value=30, value=14)

if st.button("Run Forecast"):
    model = Prophet()
    model.fit(df_res)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_res["ds"], y=df_res["y"], name="Actual Usage", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], name="Forecast", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_upper"], name="Upper Bound", line=dict(color="lightgrey"), showlegend=False))
    fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat_lower"], name="Lower Bound", line=dict(color="lightgrey"), fill='tonexty', showlegend=False))

    fig.update_layout(title=f"Forecast for {resource}", xaxis_title="Date", yaxis_title="Used Quantity", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📋 Forecast Data")
    forecast_display = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)
    forecast_display.columns = ["Date", "Predicted Usage", "Lower Estimate", "Upper Estimate"]
    st.dataframe(forecast_display)
else:
    st.info("Select a resource and click 'Run Forecast' to see predictions.")
