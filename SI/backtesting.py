import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model(resource, csv_file='lab_usage.csv'):
    # Load data
    df = pd.read_csv(csv_file, parse_dates=['Date'])
    df = df[df['Resource'] == resource].copy()
    df.sort_values('Date', inplace=True)
    
    # Convert Date to numeric day
    df['Day'] = (df['Date'] - df['Date'].min()).dt.days

    # Split data: 80% train, 20% test
    split_idx = int(len(df) * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # Train model
    model = LinearRegression()
    model.fit(train[['Day']], train['UsedQty'])

    # Predict on test
    test['Predicted'] = model.predict(test[['Day']])
    
    # Round predictions for better comparison
    test['Predicted'] = test['Predicted'].round()

    # Calculate metrics
    mae = mean_absolute_error(test['UsedQty'], test['Predicted'])
    rmse = np.sqrt(mean_squared_error(test['UsedQty'], test['Predicted']))

    print("\n📊 Forecast Accuracy Metrics")
    print(f"MAE (Mean Absolute Error): {mae:.2f}")
    print(f"RMSE (Root Mean Square Error): {rmse:.2f}")
    print("\n📉 Actual vs Predicted Usage:")
    print(test[['Date', 'UsedQty', 'Predicted']])

    return mae, rmse

# Example usage:
evaluate_model("Ethanol")
