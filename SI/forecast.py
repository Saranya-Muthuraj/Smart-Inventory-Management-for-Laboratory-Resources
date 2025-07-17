import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import timedelta

def predict_and_plot(resource_name, csv_file='lab_usage.csv'):
    # Load CSV
    df = pd.read_csv(csv_file, parse_dates=['Date'])
    
    # Filter for selected resource
    df = df[df['Resource'] == resource_name].copy()
    df.sort_values('Date', inplace=True)
    
    # Convert dates to numeric "Day" values
    df['Day'] = (df['Date'] - df['Date'].min()).dt.days

    # Train linear regression model
    model = LinearRegression()
    model.fit(df[['Day']], df['UsedQty'])

    # Predict for the next 7 days
    last_day = df['Day'].max()
    future_days = [[last_day + i] for i in range(1, 8)]
    forecast = model.predict(future_days)

    # Prepare future dates for plotting
    last_date = df['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, 8)]

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(df['Date'], df['UsedQty'], label='Historical Usage', marker='o')
    plt.plot(future_dates, forecast, label='Forecasted Usage (Next 7 Days)', linestyle='--', marker='x', color='orange')
    plt.title(f"Usage Forecast for {resource_name}")
    plt.xlabel("Date")
    plt.ylabel("Used Quantity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Return the forecast values
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Usage": forecast.astype(int)
    })
    return forecast_df

# 🔍 Example usage:
if __name__ == "__main__":
    predictions = predict_and_plot("Ethanol")  # Change to "Gloves", etc.
    print(predictions)
