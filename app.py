import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from sklearn.neighbors import KNeighborsRegressor

# Set up Streamlit title and description
st.title("Infrastructure Impact on Stock Returns Prediction")
st.markdown("""
    This app predicts the daily stock returns based on infrastructure-related features using a K-Nearest Neighbors model.
    The features used for prediction are:
    - Building Permits
    - Government Infrastructure Spending
""")

# File upload for the model
st.sidebar.header("Upload Model")
uploaded_model = st.sidebar.file_uploader("Upload your .pkl model file", type=["pkl"])

# Load the uploaded model if provided
if uploaded_model is not None:
    try:
        model = joblib.load(uploaded_model)
        model_details = {"model": "KNN", "r2_score": None, "mean_squared_error": None}
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
else:
    try:
        model = joblib.load('knn_infrastructure_prediction_model.pkl')
        model_details = joblib.load('knn_model_details.pkl')
        st.sidebar.success("Default model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading default model: {str(e)}")

# Display model performance if model is loaded
if model:
    st.subheader("Model Performance")
    if model_details.get("r2_score"):
        st.write(f"Model: {model_details['model']}")
        st.write(f"R-squared: {model_details['r2_score']:.4f}")
        st.write(f"Mean Squared Error: {model_details['mean_squared_error']:.4f}")
    else:
        st.write("Model performance details are not available.")

# User input for stock selection and prediction
st.sidebar.header("Select Stock(s)")

# List of stock tickers to choose from
stock_tickers = ['ITC.NS', 'TCS.NS', 'WIPRO.NS']

# Stock selection dropdown with multiple selection option
selected_stocks = st.sidebar.multiselect("Choose stock(s):", stock_tickers)

# Fetch stock data (adjusted close prices and returns) for selected stocks
if selected_stocks:
    for selected_stock in selected_stocks:
        stock_data = yf.download(selected_stock, start='2022-01-01', end='2024-06-30')
        stock_data['Daily_Return'] = stock_data['Adj Close'].pct_change().dropna()

        # Display stock data and returns
        st.subheader(f"Stock Data for {selected_stock}")
        st.write(stock_data[['Adj Close', 'Daily_Return']].tail(10))  # Show last 10 rows of stock data

        # Plot stock returns
        st.subheader(f"Daily Returns for {selected_stock}")
        st.line_chart(stock_data['Daily_Return'])

        # Show Infrastructure Prediction for each stock
        st.subheader(f"Infrastructure Prediction for {selected_stock}")
        
        # User input for prediction (infrastructure data) with unique keys
        building_permits = st.sidebar.number_input(f"Building Permits for {selected_stock} (in billion)", min_value=0.0, step=0.1, key=f"building_permits_{selected_stock}")
        gov_infrastructure_spending = st.sidebar.number_input(f"Government Infrastructure Spending for {selected_stock} (in billion)", min_value=0.0, step=0.1, key=f"gov_infrastructure_spending_{selected_stock}")

        # Create a DataFrame with user input
        input_data = pd.DataFrame([[building_permits, gov_infrastructure_spending]], columns=['Building_Permits', 'Government_Infrastructure_Spending'])

        # Predict using the model
        if st.sidebar.button(f"Predict for {selected_stock}"):
            if model:
                prediction = model.predict(input_data)
                st.write(f"Predicted Daily Return for {selected_stock}: {prediction[0]:.4f}")

                # Display the prediction result for the user
                st.write("""
                    The predicted daily return represents how much the stock price of the selected stock is likely to change based on the 
                    given infrastructure data. A positive return indicates an increase, while a negative return represents a decrease.
                """)
                
                # Optionally, show the top features (important to know model's prediction rationale)
                st.subheader("Model Interpretation")
                st.write("""
                    The model uses two features: **Building Permits** and **Government Infrastructure Spending**. Both of these factors 
                    are known to have an impact on the economy and the stock market, especially in the infrastructure sector.
                """)
            else:
                st.error("Model not loaded. Please upload a valid .pkl file.")
else:
    st.write("Please select at least one stock to see the data and prediction.")

# Running the app
if __name__ == "__main__":
    st.markdown("### Try entering different values for the features in the sidebar to predict the stock return.")
