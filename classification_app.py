import streamlit as st
import pandas as pd
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
        st.write(f"Best Model: {best_model_name}")
        st.write(results[best_model_name]["report"])

        # Feature Importance
        feature_importance(best_model, X_test)

        # Save model and scaler for prediction
        import joblib
        joblib.dump(best_model, "best_model.pkl")
        joblib.dump(StandardScaler().fit(data.drop(columns=[target_column])), "scaler.pkl")

# Upload Unseen Data for Prediction
unseen_file = st.file_uploader("Upload CSV for Prediction", type="csv")
if unseen_file:
    unseen_data = pd.read_csv(unseen_file)
    st.write("Unseen Data Preview:")
    st.write(unseen_data.head())

    # Load trained model and scaler
    scaler, model = load_scaler_and_model("scaler.pkl", "best_model.pkl")

    # Preprocess unseen data
    processed_unseen_data = preprocess_input_data(unseen_data, scaler)

    # Predict
    if st.button("Predict"):
        predictions = predict_unseen_data(model, processed_unseen_data)
        st.write("Predictions:")
        st.write(predictions)
