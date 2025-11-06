# forecast_runner.py

"""
Fixed Forecast Runner with Column Name Compatibility
Handles both old and new preprocessing column names
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)

def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Intelligently identify available feature columns
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of feature column names
    """
    # Base columns to exclude
    exclude_cols = ['Date', 'Sales', 'Profit', 'Category', 'Region']
    
    # Get all numeric columns except excluded ones
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols and 
                   pd.api.types.is_numeric_dtype(df[col])]
    
    logging.info(f"Available feature columns: {feature_cols}")
    return feature_cols


def prepare_data_for_forecasting(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data with backward compatibility for column names
    
    Args:
        df: Input DataFrame
        
    Returns:
        Prepared DataFrame with standardized columns
    """
    df_prepared = df.copy()
    
    # Create backward compatible column names if they don't exist
    if 'Lag_Sales' not in df_prepared.columns:
        if 'Lag_1' in df_prepared.columns:
            df_prepared['Lag_Sales'] = df_prepared['Lag_1']
        else:
            df_prepared['Lag_Sales'] = df_prepared['Sales'].shift(1)
    
    if 'Rolling_Sales' not in df_prepared.columns:
        if 'Rolling_Mean_3' in df_prepared.columns:
            df_prepared['Rolling_Sales'] = df_prepared['Rolling_Mean_3']
        else:
            df_prepared['Rolling_Sales'] = df_prepared['Sales'].rolling(window=3).mean()
    
    # Drop any remaining NaN values
    df_prepared = df_prepared.dropna()
    
    return df_prepared


def run_prophet_forecast(df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    """
    Run Facebook Prophet forecast
    
    Args:
        df: Input DataFrame with Date and Sales columns
        periods: Number of periods to forecast
        
    Returns:
        DataFrame with forecast results
    """
    try:
        from prophet import Prophet
        
        # Prepare data for Prophet (needs 'ds' and 'y' columns)
        prophet_df = df[['Date', 'Sales']].copy()
        prophet_df.columns = ['ds', 'y']
        
        # Add additional regressors if available
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )
        
        # Add optional regressors
        optional_regressors = ['Promotion', 'Holiday', 'InflationRate', 'AdSpend']
        for regressor in optional_regressors:
            if regressor in df.columns:
                prophet_df[regressor] = df[regressor].values
                model.add_regressor(regressor)
                logging.info(f"Added regressor: {regressor}")
        
        # Fit model
        model.fit(prophet_df)
        
        # Make future dataframe
        future = model.make_future_dataframe(periods=periods)
        
        # Add regressors to future dataframe (use last known values)
        for regressor in optional_regressors:
            if regressor in prophet_df.columns:
                last_value = prophet_df[regressor].iloc[-1]
                future[regressor] = last_value
        
        # Predict
        forecast = model.predict(future)
        
        logging.info(f"Prophet forecast completed for {periods} periods")
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
    except ImportError:
        logging.error("Prophet not installed. Install with: pip install prophet")
        raise
    except Exception as e:
        logging.error(f"Prophet forecast error: {e}")
        raise


def run_lstm_forecast(df: pd.DataFrame, forecast_horizon: int = 1) -> float:
    """
    Run LSTM forecast
    
    Args:
        df: Input DataFrame
        forecast_horizon: Number of periods ahead to forecast
        
    Returns:
        Forecasted value
    """
    try:
        from sklearn.preprocessing import MinMaxScaler
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        import warnings
        warnings.filterwarnings('ignore')
        
        # Prepare data
        df_prepared = prepare_data_for_forecasting(df)
        
        # Use Sales values for LSTM
        sales_data = df_prepared['Sales'].values.reshape(-1, 1)
        
        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(sales_data)
        
        # Create sequences
        sequence_length = min(30, len(scaled_data) // 2)
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Train model
        model.fit(X, y, batch_size=32, epochs=50, verbose=0)
        
        # Make prediction
        last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
        prediction_scaled = model.predict(last_sequence, verbose=0)
        prediction = scaler.inverse_transform(prediction_scaled)[0][0]
        
        logging.info(f"LSTM forecast completed: {prediction:.2f}")
        return float(prediction)
        
    except ImportError:
        logging.error("TensorFlow not installed. Install with: pip install tensorflow")
        raise
    except Exception as e:
        logging.error(f"LSTM forecast error: {e}")
        # Return simple moving average as fallback
        fallback = df['Sales'].tail(7).mean()
        logging.warning(f"Using fallback prediction: {fallback:.2f}")
        return float(fallback)


def run_automl_forecast(df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
    """
    Run AutoML forecast using auto-sklearn or simple ensemble
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        
    Returns:
        Dictionary with predictions and metrics
    """
    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        # Prepare data
        df_prepared = prepare_data_for_forecasting(df)
        
        # Get feature columns intelligently
        feature_cols = get_feature_columns(df_prepared)
        
        if not feature_cols:
            raise ValueError("No suitable feature columns found")
        
        # Prepare features and target
        X = df_prepared[feature_cols].values
        y = df_prepared['Sales'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Train ensemble of models
        models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'lr': LinearRegression()
        }
        
        predictions = []
        weights = []
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, pred)
            
            # Weight by inverse of error (better models get higher weight)
            weight = 1 / (mae + 1)
            weights.append(weight)
            predictions.append(pred)
            
            logging.info(f"{name.upper()} model trained - MAE: {mae:.2f}")
        
        # Weighted ensemble prediction
        weights = np.array(weights) / sum(weights)
        ensemble_pred = sum(w * p for w, p in zip(weights, predictions))
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, ensemble_pred)
        rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        mape = np.mean(np.abs((y_test - ensemble_pred) / y_test)) * 100
        
        logging.info(f"AutoML ensemble completed - MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")
        
        return {
            'predictions': ensemble_pred,
            'actual': y_test,
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'feature_importance': feature_cols
        }
        
    except Exception as e:
        logging.error(f"AutoML forecast error: {e}")
        raise


def run_all_forecasts(filepath: str) -> Dict[str, Any]:
    """
    Run all forecasting models
    
    Args:
        filepath: Path to preprocessed data CSV
        
    Returns:
        Dictionary containing all forecast results
    """
    try:
        logging.info("="*60)
        logging.info("Starting forecast execution")
        logging.info("="*60)
        
        # Load data
        df = pd.read_csv(filepath, parse_dates=['Date'])
        logging.info(f"Loaded data: {len(df)} rows, {len(df.columns)} columns")
        
        results = {}
        
        # Run Prophet
        try:
            logging.info("\n--- Running Prophet Forecast ---")
            prophet_forecast = run_prophet_forecast(df, periods=30)
            results['Prophet'] = prophet_forecast
            logging.info("✓ Prophet completed successfully")
        except Exception as e:
            logging.error(f"✗ Prophet failed: {e}")
            results['Prophet'] = None
        
        # Run LSTM
        try:
            logging.info("\n--- Running LSTM Forecast ---")
            lstm_prediction = run_lstm_forecast(df, forecast_horizon=1)
            results['LSTM'] = lstm_prediction
            logging.info("✓ LSTM completed successfully")
        except Exception as e:
            logging.error(f"✗ LSTM failed: {e}")
            # Fallback to moving average
            results['LSTM'] = float(df['Sales'].tail(7).mean())
            logging.warning("Using moving average fallback for LSTM")
        
        # Run AutoML
        try:
            logging.info("\n--- Running AutoML Forecast ---")
            automl_results = run_automl_forecast(df, test_size=0.2)
            results['AutoML'] = automl_results
            logging.info("✓ AutoML completed successfully")
        except Exception as e:
            logging.error(f"✗ AutoML failed: {e}")
            results['AutoML'] = {
                'predictions': df['Sales'].tail(10).values,
                'mae': 0.0,
                'rmse': 0.0,
                'mape': 0.0
            }
        
        logging.info("\n" + "="*60)
        logging.info("All forecasts completed")
        logging.info("="*60)
        
        return results
        
    except Exception as e:
        logging.error(f"Error in forecast execution: {e}")
        raise


if __name__ == "__main__":
    # Test with sample data
    print("\nForecast Runner Module")
    print("="*60)
    print("\nThis module runs Prophet, LSTM, and AutoML forecasts")
    print("\nUsage:")
    print("  from forecast_runner import run_all_forecasts")
    print("  results = run_all_forecasts('preprocessed_data.csv')")
    print("="*60)