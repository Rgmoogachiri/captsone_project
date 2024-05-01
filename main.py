import streamlit as st
import yfinance as yf
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import base64
import pandas as pd

# Define constants
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
STOCKS = ('TSLA', 'AMZN', 'BRK.B', 'FB', 'MSFT', 'NVDA', 'CRM', 'GOOG')

# Function to load data
@st.cache_data
def load_data(ticker):
    try:
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

# Function to plot raw data
def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Stock Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Stock Close"))
    fig.layout.update(title_text='Time Series Rangeslider', xaxis_rangeslider_visible=False)
    return fig

# Function to get image as base64
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def get_latest_prices(data):
    latest_date = data.iloc[-1]['Date']
    latest_price = data.iloc[-1]['Close']
    return latest_date,latest_price




# Main function
def main():

        # Background image styling
    img = get_img_as_base64("image.avif")
    page_bg_img = f"""
            <style>
                [data-testid="stAppViewContainer"] > .main {{
                    background-image: url("data:image/png;base64,{img}");
                    background-size: cover;
                    background-position: top right;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                }}
            </style>
        """
    st.markdown(page_bg_img, unsafe_allow_html=True)
    st.title('Stock Forecasting')

    # Sidebar for additional options
    st.sidebar.subheader('Download data')

    # Load data and display loading message
    selected_stock = st.sidebar.selectbox('Select dataset for prediction', STOCKS)
    data_load_state = st.text('Loading data...')
    data = load_data(selected_stock)
    if data is not None:
        st.success('Data loaded successfully!')
        # Plot raw data
        st.subheader(f"Raw Data Plot: {selected_stock}")
        fig = plot_raw_data(data)
        Closing_date,latest_closing_price = get_latest_prices(data)
        # st.write("## Latest Prices")
        # col1, col2 = st.columns(2)
        # with col1:
        # st.write("Closing Price:", Closing_date,latest_closing_price)
        # with col2:
            # st.write("Closing Price:", latest_closing_price)
        # st.write("Latest Closing Date:", latest_date)
        # st.write(f"Today's Opening Price for: {selected_stock}", today_opening_price)
        st.plotly_chart(fig)

        # Forecasting
        st.sidebar.subheader('Forecasting Options')
        n_years = st.sidebar.slider('Years of prediction:', 1, 4)
        period = n_years * 365

        # Prepare data for forecasting
        df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})

        # Prophet model fitting and forecasting
        m = Prophet(interval_width=0.95)
        m.fit(df_train)
        future = m.make_future_dataframe(periods=period, include_history=True)
        forecast = m.predict(future)

        # Display forecast components
        st.subheader('Forecast Components')
        st.write(forecast.tail())
        fig2 = m.plot_components(forecast)
        st.write(fig2)


# Run the app
if __name__ == "__main__":
    main()
