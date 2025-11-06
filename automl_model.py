# automl_model.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
import logging

def run_automl(df):
    try:
        # ✅ Required features
        base_features = ['Month', 'Year', 'DayOfWeek', 'Lag_Sales', 'Rolling_Sales']
        
        # ✅ Optional external factors
        optional_features = ['InflationRate', 'RealGDPGrowth']
        available_features = [col for col in optional_features if col in df.columns]
        
        # ✅ Combine available features
        selected_features = base_features + available_features
        X = df[selected_features]
        y = df['Sales']

        # ✅ Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

        # ✅ Train model
        model = RandomForestRegressor()
        model.fit(X_train, y_train)

        # ✅ Predict and evaluate
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)

        logging.info(f"AutoML model trained with features: {selected_features}")
        logging.info(f"AutoML MAE: {mae:.2f}")

        return preds, mae

    except Exception as e:
        logging.error(f"AutoML model error: {e}")
        raise