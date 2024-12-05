import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objs as go
import yfinance as yf
from datetime import datetime, timedelta

# Function to load the model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('gold_prediction_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to fetch real-time and historical gold prices using yfinance
def fetch_gold_data(period='1mo'):
    try:
        # Fetch gold futures data (Gold Continuous Contract)
        gold = yf.Ticker('GC=F')
        
        # Get current price using a more robust method
        current_price = None
        try:
            current_price = gold.fast_info['last_price']
        except Exception:
            try:
                current_price = gold.history(period='1d')['Close'].iloc[-1]
            except Exception as e:
                st.error(f"Could not retrieve current price: {e}")
        
        if current_price is None:
            st.error("Unable to fetch current gold price")
            return None, None
        
        # Get historical data for the selected period
        historical_data = gold.history(period=period)
        
        if historical_data.empty:
            st.error("No historical data available")
            return current_price, None
        
        # Prepare historical data DataFrame
        historical_df = historical_data.reset_index()
        historical_df['Date'] = historical_df['Date'].dt.strftime('%Y-%m-%d')
        historical_df['Change'] = historical_df['Close'].pct_change() * 100
        
        # Select and rename columns
        gold_historical = historical_df[['Date', 'Close', 'Open', 'High', 'Low', 'Change']].rename(
            columns={'Close': 'Price'}
        )
        
        return current_price, gold_historical
    
    except Exception as e:
        st.error(f"Error fetching gold data: {e}")
        return None, None

# Backup method to fetch gold price
def fetch_backup_gold_price():
    try:
        # List of alternative tickers to try
        tickers = ['GOLD', 'GLD', 'IAU']
        
        for ticker in tickers:
            try:
                gold = yf.Ticker(ticker)
                price = gold.fast_info['last_price']
                if price:
                    return price
            except Exception:
                continue
        
        # If no price found
        st.error("Could not fetch gold price from alternative sources")
        return None
    
    except Exception as e:
        st.error(f"Backup price fetch error: {e}")
        return None

# Function to preprocess the data
def preprocess_data(data):
    try:
        # Ensure numeric columns
        numeric_columns = ['Price', 'Open', 'High', 'Low', 'Change']
        for col in numeric_columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Drop any rows with NaN values
        data = data.dropna()
        
        return data
    except Exception as e:
        st.error(f"Data preprocessing error: {e}")
        return None

# Function to predict the next gold price based on the model
def predict_next_price(model, last_data, scaler):
    try:
        # Make sure last_data has 5 features: ['Price', 'Open', 'High', 'Low', 'Change']
        last_data_scaled = scaler.transform(last_data)

        # Reshape the input to match the expected shape (1, 5, 5)
        input_sequence = last_data_scaled.reshape(1, 5, 5)
        
        # Predict the next price
        prediction = model.predict(input_sequence)
        
        # Inverse transform the prediction to get the original scale
        return scaler.inverse_transform(prediction)[0][3]  # Assuming 'Price' is at index 3
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Main function for the app
def main():
    st.title('Gold Price Prediction App')

    # User input for period selection
    period = st.selectbox("Select Period for Historical Data", 
                          ['5d', '1wk', '1mo', '3mo', '6mo', '1y', '5y'])

    # Fetch current and historical gold prices based on the selected period
    current_price, historical_data = fetch_gold_data(period)

    # If primary method fails, try backup
    if current_price is None:
        current_price = fetch_backup_gold_price()

    if current_price is not None:
        st.metric('Current Gold Price (USD)', f'${current_price:.2f}')

        # If historical data wasn't fetched, create dummy data
        if historical_data is None:
            st.warning("Using simulated historical data")
            date_range = pd.date_range(datetime.today() - pd.Timedelta(days=30), periods=30).strftime('%Y-%m-%d')
            historical_prices = np.random.uniform(current_price * 0.9, current_price * 1.1, size=30)
            
            historical_data = pd.DataFrame({
                'Date': date_range,
                'Price': historical_prices,
                'Open': historical_prices * 0.98,
                'High': historical_prices * 1.02,
                'Low': historical_prices * 0.97,
                'Change': np.random.uniform(-2, 2, size=30)
            })

        # Preprocess the data
        historical_data = preprocess_data(historical_data)

        if historical_data is not None and not historical_data.empty:
            # Create candlestick chart
            fig = go.Figure(data=[go.Candlestick(
                x=historical_data['Date'],
                open=historical_data['Open'],
                high=historical_data['High'],
                low=historical_data['Low'],
                close=historical_data['Price']
            )])
            st.plotly_chart(fig)

            # Prediction Section
            st.subheader('Price Prediction')
            if st.button('Predict Next Price'):
                # Use MinMaxScaler on features
                features = ['Open', 'High', 'Low', 'Price', 'Change']
                scaler = MinMaxScaler()
                scaler.fit(historical_data[features])

                # Prepare the last 5 days of data for prediction
                last_5_days = historical_data[features].tail(5)

                # Load the model
                model = load_model()
                if model is None:
                    st.error("Failed to load prediction model")
                    return

                # Predict the next gold price
                predicted_price = predict_next_price(model, last_5_days, scaler)

                if predicted_price is not None:
                    st.metric('Predicted Next Gold Price', f'${predicted_price:.2f}')
                else:
                    st.error("Price prediction failed")
        else:
            st.error("Could not process historical gold prices")
    else:
        st.error("Could not fetch gold price data from any source")

if __name__ == '__main__':
    main()
