**Smart Lab Inventory System**

A voice- and data-enabled Streamlit application for managing laboratory inventory, forecasting resource demand, and ensuring timely restocking and expiry tracking. Built with Python, Prophet, Plotly, and Google Speech Recognition.

**Features**

**Voice-Controlled Inventory Management**
- Use voice commands (e.g., “Use 5 ml of ethanol”) to manage inventory in real-time.
- Automatic updates to inventory and usage logs.

**Demand Forecasting**
- Forecast resource usage for the next 7–30 days using the Facebook Prophet model.
- Interactive Plotly charts with upper and lower prediction bounds.

**Inventory Logs**
- View the latest voice/manual interactions with the system.
- Tracks both “use” and “return” actions.

**Restocking Alerts**
- Highlights inventory items that fall below a set threshold (default: 10 units).

**Expiry Monitoring**
- Automatically detects resources expiring within the next 30 days.


**File Structure**

smart-lab-inventory/
├── lab_inventory.csv         # Initial inventory data
├── lab_usage.csv             # Logs of usage actions (created automatically)
├── lab_usage1.csv            # Resource usage history for forecasting
├── app.py                    # Main Streamlit application script
├── README.md                 # Project documentation (this file)

**Required Libraries**
pip install streamlit pandas prophet plotly SpeechRecognition

**Running the App**
streamlit run app.py

**Example Voice Commands**
Use 10 ml of ethanol
Return 5 units of gloves
