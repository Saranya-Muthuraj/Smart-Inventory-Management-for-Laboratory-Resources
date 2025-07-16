import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta
import numpy as np
import matplotlib.pyplot as plt

def rolling_forecast_accuracy(resource_name, csv_file='lab_usage.csv', train_days=30, forecast_days=7):
    # Load and filter data
    df = pd.read_csv(csv_file, parse_dates=['Date'])
    df = df[df['Resource'] == resource_name].copy()
    df.sort_values('Date', inplace=True)

    # Get cutoff dates
    latest_date = df['Date'].max()
    start_date = latest_date - timedelta(days=train_days + forecast_days)

    # Use recent history for both training and testing
    recent_df = df[df['Date'] >= start_date].copy()

    # Split into training and test
    train_df = recent_df.iloc[:-forecast_days]
    test_df = recent_df.iloc[-forecast_days:]

    # Format for Prophet
    prophet_train = train_df[['Date', 'UsedQty']].rename(columns={'Date': 'ds', 'UsedQty': 'y'})

    # Train Prophet
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_train)

    # Forecast into test period
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # Get forecast for the test period only
    forecast = forecast[['ds', 'yhat']].set_index('ds')
    test_df = test_df.set_index('Date')
    test_df['Predicted'] = forecast.loc[test_df.index]['yhat'].round()

    # Calculate accuracy
    mae = mean_absolute_error(test_df['UsedQty'], test_df['Predicted'])
    rmse = np.sqrt(mean_squared_error(test_df['UsedQty'], test_df['Predicted']))

    # Print results
    print("\n📊 Rolling Forecast Accuracy (Last 30 Days Training)")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print("\n📉 Actual vs Predicted:")
    print(test_df[['UsedQty', 'Predicted']])

    # Plot
    test_df[['UsedQty', 'Predicted']].plot(title=f"{resource_name} - Actual vs Predicted (Rolling 30 Days)")
    plt.xlabel("Date")
    plt.ylabel("Usage")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return mae, rmse

# 🔍 Example usage
if __name__ == "__main__":
    rolling_forecast_accuracy("Ethanol")
