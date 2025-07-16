import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

def forecast_with_prophet(resource_name, csv_file='lab_usage.csv'):
    # Load data
    df = pd.read_csv(csv_file, parse_dates=['Date'])

    # Filter by resource
    df = df[df['Resource'] == resource_name].copy()
    df.sort_values('Date', inplace=True)

    # Prepare for Prophet: rename columns
    prophet_df = df[['Date', 'UsedQty']].rename(columns={'Date': 'ds', 'UsedQty': 'y'})

    # Create and train Prophet model
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)

    # Create future DataFrame (7 days ahead)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    # Plot forecast
    model.plot(forecast)
    plt.title(f"Forecasted Usage for {resource_name}")
    plt.xlabel("Date")
    plt.ylabel("Usage")
    plt.tight_layout()
    plt.grid(True)
    plt.show()

    # Show only forecasted part
    result = forecast[['ds', 'yhat']].tail(7)
    result = result.rename(columns={'ds': 'Date', 'yhat': 'Predicted_Usage'})
    result['Predicted_Usage'] = result['Predicted_Usage'].astype(int)
    return result

# 🔍 Example usage
if __name__ == "__main__":
    result = forecast_with_prophet("Ethanol")  # Change to any resource
    print("\n📊 7-Day Forecast:")
    print(result)
