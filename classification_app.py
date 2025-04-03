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

st.set_page_config(layout="wide")
st.title("🧠 Machine Learning Pipeline App")

# === Upload Dataset for Training ===
uploaded_file = st.file_uploader("📁 Upload CSV for Training", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📊 Preview of Uploaded Data")
    st.write(data.head())

    # === Preprocessing ===
    data = handle_missing_values(data)
    data = handle_outliers(data)

    # === Exploratory Data Analysis ===
    target_column = st.sidebar.selectbox("🎯 Select Target Column", data.columns)
    explore_data(data, target_column)

    # === Model Training ===
    if st.sidebar.button("🚀 Train Models"):
        best_models, X_test, y_test = train_models(data, target_column)
        best_model_name, best_model, results = evaluate_model(best_models, X_test, y_test)
        
        st.subheader(f"✅ Best Model: {best_model_name}")
        
        st.subheader("📋 Classification Report")
        st.json(results[best_model_name]["report"])

        # === Show Best Hyperparameters ===
        if hasattr(best_model, "best_params_"):
            st.subheader("🔧 Best Hyperparameters (from GridSearchCV)")
            st.json(best_model.best_params_)
        elif hasattr(best_model, "get_params"):
            st.subheader("🔧 Model Parameters")
            st.json(best_model.get_params())

        # === Feature Importance ===
        st.subheader("📌 Feature Importances")
        feature_df = feature_importance(best_model, X_test)
        st.dataframe(feature_df)

        # === Save Model and Scaler ===
        scaler = StandardScaler().fit(data.drop(columns=[target_column]))
        joblib.dump(best_model, "best_model.pkl")
        joblib.dump(scaler, "scaler.pkl")

# === Upload Unseen Data for Prediction ===
unseen_file = st.file_uploader("📁 Upload CSV for Prediction", type="csv")

if unseen_file:
    unseen_data_


