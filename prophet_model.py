#prophet_model.py

from prophet import Prophet
import pandas as pd
import logging

def run_prophet(df):
    try:
        df_prophet = df[['Date', 'Sales']].rename(columns={'Date': 'ds', 'Sales': 'y'})
        model = Prophet()
        model.add_country_holidays(country_name='IN')
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=12, freq='MS')
        forecast = model.predict(future)
        logging.info("Prophet model forecast generated.")
        return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    except Exception as e:
        logging.error(f"Prophet model error: {e}")
        raise