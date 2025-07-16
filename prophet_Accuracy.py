import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

def prophet_forecast_accuracy(resource_name, csv_file='lab_usage.csv', test_days=7):
    # Load data
    df = pd.read_csv(csv_file, parse_dates=['Date'])
    df = df[df['Resource'] == resource_name].copy()
    df.sort_values('Date', inplace=True)

    # Rename for Prophet
    prophet_df = df[['Date', 'UsedQty']].rename(columns={'Date': 'ds', 'UsedQty': 'y'})

    # Split into train and test
    train_df = prophet_df[:-test_days]
    test_df = prophet_df[-test_days:]

    # Train model
    model = Prophet(daily_seasonality=True)
    model.fit(train_df)

    # Forecast same length as test set
    future = model.make_future_dataframe(periods=test_days)
    forecast = model.predict(future)

    # Get only predicted values matching test dates
    predicted = forecast[['ds', 'yhat']].set_index('ds').loc[test_df['ds']]

    # Combine with actual values
    test_df = test_df.set_index('ds')
    test_df['Predicted'] = predicted['yhat']
    test_df['Predicted'] = test_df['Predicted'].round()

    # Metrics
    mae = mean_absolute_error(test_df['y'], test_df['Predicted'])
    rmse = np.sqrt(mean_squared_error(test_df['y'], test_df['Predicted']))

    print("\n📊 Prophet Forecast Accuracy")
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Square Error): {rmse:.2f}")
    print("\n📉 Actual vs Predicted:")
    print(test_df[['y', 'Predicted']])

    # Plot
    test_df[['y', 'Predicted']].plot(title=f"{resource_name} - Actual vs Predicted")
    plt.xlabel("Date")
    plt.ylabel("Usage")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return mae, rmse

# 🔍 Example usage
if __name__ == "__main__":
    prophet_forecast_accuracy("Ethanol")
