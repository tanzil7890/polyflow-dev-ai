import pandas as pd
import numpy as np
import polyflow
from polyflow.models.language_temporal import TemporalLanguageProcessor
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def setup_test_data():
    """Create synthetic time series data with known patterns and anomalies"""
    # Generate dates for one year of daily data
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Generate base signal with multiple seasonal patterns
    t = np.arange(len(dates))
    weekly_pattern = 3 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
    monthly_pattern = 5 * np.sin(2 * np.pi * t / 30)  # Monthly seasonality
    quarterly_pattern = 8 * np.sin(2 * np.pi * t / 90)  # Quarterly seasonality
    
    # Add gradual trend with changing slope
    trend = 0.01 * t + 0.00005 * t**2
    
    # Add random noise with varying volatility
    base_noise = np.random.normal(0, 0.8, len(dates))
    # Higher volatility during summer months
    summer_mask = (dates.month >= 6) & (dates.month <= 8)
    base_noise[summer_mask] *= 2
    
    # Combine components
    values = 20 + weekly_pattern + monthly_pattern + quarterly_pattern + trend + base_noise
    
    # Insert various artificial anomalies
    # Sharp spike anomalies
    values[50] += 15  # Single day spike
    values[150:152] += 12  # Two-day spike
    
    # Dip anomalies
    values[200:203] -= 10  # Three-day dip
    
    # Level shift anomalies
    values[300:350] *= 1.3  # Step increase for 50 days
    
    # Trend change anomaly
    for i in range(100):
        values[400 + i] += 0.2 * i  # Accelerated trend for 100 days
    
    # Seasonal pattern break
    seasonal_break_start = 250
    seasonal_break_end = 280
    values[seasonal_break_start:seasonal_break_end] = (
        values[seasonal_break_start:seasonal_break_end] - monthly_pattern[seasonal_break_start:seasonal_break_end]
    )
    
    # Create missing values (NaN) for some periods
    missing_indices = np.random.choice(range(len(dates)), 15, replace=False)
    values[missing_indices] = np.nan
    
    return pd.DataFrame({
        'timestamp': dates,
        'value': values
    })

def test_anomaly_detection():
    """Test anomaly detection functionality with the enhanced dataset"""
    # Setup
    api_key = "OPEN_AI_API_KEY"
    
    # Initialize with time series specific settings
    lm = TemporalLanguageProcessor(
        model_identifier="gpt-4o",
        api_credentials=api_key,
        generation_temperature=0.1,
        response_max_length=2048
    )
    polyflow.settings.configure(lm=lm)
    
    # Get test data
    df = setup_test_data()
    
    try:
        # Detect anomalies with proper message format
        anomalies = df.vecsem_time_series.detect_anomalies(
            time_col="timestamp",
            value_col="value",
            description="Identify anomalies including sudden spikes, drops, level shifts, trend changes, and seasonal pattern breaks",
            threshold=0.75,
            safe_mode=True
        )
        
        if anomalies is not None and not anomalies.empty:
            print("\nAnomaly Detection Results:")
            print("=" * 80)
            detected = anomalies[anomalies['is_anomaly']].sort_values('anomaly_score', ascending=False)
            print(f"Found {len(detected)} anomalies")
            print(detected[['timestamp', 'value', 'anomaly_score', 'anomaly_explanation']].head(10))
            return anomalies
        else:
            print("No anomalies detected")
            return None
    except Exception as e:
        print(f"Error in anomaly detection: {str(e)}")
        return None

def test_forecasting():
    """Test forecasting functionality with confidence intervals"""
    df = setup_test_data()
    
    try:
        # Generate 30-day forecast with confidence intervals
        forecast = df.vecsem_time_series.forecast(
            time_col="timestamp",
            value_col="value",
            horizon=30,
            context="Consider weekly, monthly, and quarterly seasonal patterns and the gradual upward trend",
            confidence_intervals=True
        )
        
        if forecast is not None and not forecast.empty:
            print("\nForecast Results:")
            print("=" * 80)
            print(f"Generated {len(forecast)} days of forecasts")
            print(forecast.head(10))
            
            # Calculate forecast accuracy metrics
            last_date = df['timestamp'].max()
            validation_start = last_date - timedelta(days=30)
            validation_data = df[df['timestamp'] >= validation_start]
            
            print("\nForecast Validation (Last 30 days):")
            print(f"Number of validation points: {len(validation_data)}")
            return forecast
        else:
            print("No forecast generated")
            return None
    except Exception as e:
        print(f"Error in forecasting: {str(e)}")
        return None

def main():
    """Run time series analysis tests"""
    print("Initializing time series analysis tests...")
    print("\nGenerating Synthetic Time Series Data:")
    df = setup_test_data()
    
    # Print data summary
    print(f"Data Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Number of observations: {len(df)}")
    print(f"Number of missing values: {df['value'].isna().sum()}")
    print("\nData Statistics:")
    print(df['value'].describe())
    
    print("\nRunning Anomaly Detection Test...")
    anomalies = test_anomaly_detection()
    
    if anomalies is not None:
        print("\nRunning Forecasting Test...")
        forecast = test_forecasting()
        
        if forecast is not None:
            try:
                # Print LM usage
                polyflow.settings.lm.display_usage_summary()
            except Exception as e:
                print(f"Error printing usage: {str(e)}")
    
    print("\nTime series analysis tests completed.")

if __name__ == "__main__":
    main()
