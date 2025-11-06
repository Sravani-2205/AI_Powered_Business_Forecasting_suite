#data_preprocessing.py

def preprocess_data(filepath):
    try:
        df = pd.read_csv(filepath, parse_dates=['Date'], dayfirst=True)
        df = df.sort_values('Date')
        df['Month'] = df['Date'].dt.month
        df['Year'] = df['Date'].dt.year
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        
        # Create exactly these column names
        df['Lag_Sales'] = df['Sales'].shift(1)
        df['Rolling_Sales'] = df['Sales'].rolling(window=3).mean()
        
        df = df.dropna()
        logging.info("Data preprocessing completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error in preprocessing: {e}")
        raise