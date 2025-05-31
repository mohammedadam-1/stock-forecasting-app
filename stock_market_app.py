import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.graph_objects as go 
import plotly.express as px 
import datetime 
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller

st.set_page_config(
    page_title=" Stock Forecasting App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #1f4e79, #2e8b57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f4e79;
        margin: 0.5rem 0;
    }
    
    .stAlert {
        border-radius: 10px;
    }
    
    /* Mobile responsive */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .stColumns > div {
            width: 100% !important;
            margin-bottom: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)


app_name = ("Welcome to Stock Forecasting App")
st.title(app_name)
st.subheader("Now you can forecast share prices of your favorite stocks without any hassle.")
#adding an image from online resources
st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTznVQzgg7i_Qep8ioqzq77z6m_pnolynhMeQ&s")
st.markdown("---")
st.header("Your selected data for forecasting")

# Take input from the user about the start and end date
# sidebar
st.sidebar.header("Select the parameters from below")

start_date = st.sidebar.date_input("Start date", date(2020, 1, 1))
end_date = st.sidebar.date_input("End date", date(2020, 12, 31))

# Add ticker list for selecting companies
ticker_list = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "FB", "NFLX", "NVDA", "BRK-B", "V", "JPM", "UNH", "PG", "HD"]
ticker = st.sidebar.selectbox("Select the company", ticker_list)

data = yf.download(ticker, start=start_date, end=end_date)
data.insert(0, "Date", data.index, True)
data.reset_index(drop=True, inplace=True)
st.write("Data From", start_date, "to", end_date)
st.write(data)

# Plot the data

st.header("Data Visualization")

# Flatten the MultiIndex columns

data.columns = [f"{col[0]}" if col[1] else col[0] for col in data.columns]
plot_columns = [col for col in data.columns if col != "Date"]
fig = px.line(data, x="Date", y=plot_columns, title="Closing price of a stock", width=1000, height=600)
st.plotly_chart(fig)

# Add a select box to select columns for forecasting
column = st.selectbox("Select column to forecast the price", plot_columns)

# subsetting the data
data = data[["Date", column]]
st.write("Selected data columns")
st.write(data)


# Check if the data is Stationary?
st.header("Is data stationary?")

# ADF Test for checking data stationarity
st.write(adfuller(data[column])[1] < 0.05)

# Decompose the data
st.header("Decomposition of the data")
decomposition = seasonal_decompose(data[column], model="additive", period=12)
st.write(decomposition.plot())

# make the same plots using plotly
st.header("Seasonal, Trend and Residual Plots")
st.plotly_chart(px.line(data, x="Date", y=decomposition.trend, title="Trend", width=1000, height=400, labels={"x": "Date", "y": "Price"}).update_traces(line_color="Green"))
st.plotly_chart(px.line(data, x="Date", y=decomposition.seasonal, title="Seasonal", width=1000, height=400, labels={"X": "Date", "y": "Price"}).update_traces(line_color="Red"))
st.plotly_chart(px.line(data, x="Date", y=decomposition.resid, title="Residuals", width=1000, height=400, labels={"x": "Date", "y": "Price"}).update_traces(line_color="Purple"))


# Take user input for model training

p = st.slider("Select the p value", 0, 5, 2)
d = st.slider("Select the d value", 0, 5, 1)
q = st.slider("Select the q value", 0, 5, 2)
seasonal_order = st.slider("Select the seasonal period", 0, 24, 12)

# train the model

model = sm.tsa.statespace.SARIMAX(data[column], order=(p,d,q), seasonal_order=(p,d,q,seasonal_order))
model = model.fit()

# print model summary
st.header("Model Summary")
st.write(model.summary())
st.write("---")

# run the model
forecast_time = st.number_input("Enter the days to be forecasted", 0, 365, 10)
predictions = model.get_prediction(start=len(data), end=len(data)+forecast_time)
predictions = predictions.predicted_mean

predictions.index = pd.date_range(start=end_date, periods=len(predictions), freq="D")
predictions = pd.DataFrame(predictions)
predictions.insert(0, "Date", predictions.index, True)
predictions.reset_index(drop=True, inplace=True)
st.write("Predictions", predictions)
st.write("Actual data", data)

# lets plot the data

fig = go.Figure()
# add actual data to the plot
fig.add_trace(go.Scatter(x=data["Date"], y=data[column], mode="lines", name="Actual", line=dict(color="blue")))
# add predicted data to the plot
fig.add_trace(go.Scatter(x=predictions["Date"], y=predictions["predicted_mean"], mode="lines", name="Predicted", line=dict(color="red")))
# set the title and axis labels   
fig.update_layout(title="Actual vs Predicted", xaxis_title="Date", yaxis_title="Price", width=1200, height=400)
st.plotly_chart(fig)   
 
       
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    Created by Mohammed Adam | 
    <a href="https://github.com/yourusername" target="_blank">GitHub</a> | 
    <a href="https://www.linkedin.com/in/mohammed-adam-aaa5621b5/" target="_blank">LinkedIn</a> | 
    <a href="mail to: mohammedadam@gmail.com">mail</a>
    </div>
    """, 
    unsafe_allow_html=True
)        