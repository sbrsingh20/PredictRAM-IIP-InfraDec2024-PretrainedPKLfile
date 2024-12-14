import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsRegressor
from io import BytesIO

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
    # Load the model from the uploaded file
    try:
        model = joblib.load(uploaded_model)
        model_details = {"model": "KNN", "r2_score": None, "mean_squared_error": None}
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {str(e)}")
else:
    # Default to the pre-saved model if no file is uploaded
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

# User input for prediction
st.sidebar.header("Input Features")

# Getting input from the user
building_permits = st.sidebar.number_input("Building Permits (in billion)", min_value=0.0, step=0.1)
gov_infrastructure_spending = st.sidebar.number_input("Government Infrastructure Spending (in billion)", min_value=0.0, step=0.1)

# Create a DataFrame with user input
input_data = pd.DataFrame([[building_permits, gov_infrastructure_spending]], columns=['Building_Permits', 'Government_Infrastructure_Spending'])

# Predict using the model
if st.sidebar.button("Predict"):
    if model:
        prediction = model.predict(input_data)
        st.subheader("Prediction")
        st.write(f"Predicted Daily Return: {prediction[0]:.4f}")

        # Additional details about the prediction
        st.write("""
            The predicted daily return represents how much the stock price of ITC is likely to change based on the 
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
        
# Running the app
if __name__ == "__main__":
    st.markdown("### Try entering different values for the features in the sidebar to predict the stock return.")
