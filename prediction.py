import pandas as pd
from sklearn.preprocessing import StandardScaler
import streamlit as st

def preprocess_input_data(data, trained_scaler=None):
    """
    Preprocess the unseen data before prediction.
    """
    # Check for missing values
    if data.isnull().sum().sum() > 0:
        st.warning("Missing values detected. Filling with column means.")
        data = data.fillna(data.mean())
    
    # Standardize the data if a scaler is provided
    if trained_scaler:
        st.write("Standardizing the input data...")
        data = pd.DataFrame(trained_scaler.transform(data), columns=data.columns)
    
    return data

def predict_unseen_data(model, input_data):
    """
    Predict the target for unseen data using the trained model.
    """
    predictions = model.predict(input_data)
    return predictions

def load_scaler_and_model(scaler_path, model_path):
    """
    Load the trained scaler and model from files (if saved as pickle or joblib).
    """
    import joblib
    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    return scaler, model
