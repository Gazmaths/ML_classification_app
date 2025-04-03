import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

from preprocess import handle_missing_values, handle_outliers
from explore import explore_data
from train import train_models
from model import evaluate_model, feature_importance
from prediction import preprocess_input_data, predict_unseen_data, load_scaler_and_model

st.title("Machine Learning Pipeline App")

# Upload Dataset
uploaded_file = st.file_uploader("Upload CSV for Training", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Preview of Uploaded Data:")
    st.write(data.head())

    # Preprocessing
    data = handle_missing_values(data)
    data = handle_outliers(data)

    # Exploratory Data Analysis
    target_column = st.sidebar.selectbox("Select Target Column", data.columns)
    explore_data(data, target_column)

    # Model Training
    if st.sidebar.button("Train Models"):
        best_models, X_test, y_test = train_models(data, target_column)
        best_model_name, best_model, results = evaluate_model(best_models, X_test, y_test)
        
        st.write(f"✅ **Best Model:** {best_model_name}")
        st.write("📊 **Classification Report:**")
        st.json(results[best_model_name]["report"])

        # Feature Importance (Random Forest style)
        st.write("📌 **Feature Importances:**")
        feature_df = feature_importance(best_model, X_test)
        st.dataframe(feature_df)

        # Save model and scaler for later prediction
        scaler = StandardScaler().fit(data.drop(columns=[target_column]))
        joblib.dump(best_model, "best_model.pkl")
        joblib.dump(scaler, "scaler.pkl")

# Upload Unseen Data for Prediction
unseen_file = st.file_uploader("Upload CSV for Prediction", type="csv")
if unseen_file:
    unseen_data = pd.read_csv(unseen_file)
    st.write("📄 **Unseen Data Preview:**")
    st.write(unseen_data.head())

    # Load model and scaler
    scaler, model = load_scaler_and_model("scaler.pkl", "best_model.pkl")

    # Preprocess and Predict
    processed_unseen_data = preprocess_input_data(unseen_data, scaler)

    if st.button("Predict"):
        predictions = predict_unseen_data(model, processed_unseen_data)
        st.write("🧠 **Predictions:**")
        st.write(predictions)

