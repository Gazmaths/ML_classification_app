import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import streamlit as st

def handle_missing_values(data):
    if st.sidebar.checkbox("Check for Missing Values"):
        st.write("Missing Values in Each Column:")
        st.write(data.isnull().sum())
        option = st.sidebar.selectbox("Handle Missing Values", ["None", "Fill with Mean", "Drop Rows"])
        if option == "Fill with Mean":
            imputer = SimpleImputer(strategy="mean")
            data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
        elif option == "Drop Rows":
            data = data.dropna()
    return data

def handle_outliers(data):
    if st.sidebar.checkbox("Check for Outliers"):
        st.write("Outliers are identified using Z-Score:")
        
        # Calculate Z-Score
        z = np.abs((data - data.mean()) / data.std())
        
        # Display Z-Scores
        st.write("Z-Score for the dataset:")
        st.write(z)
        
        # Set Z-Score threshold
        threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 5.0, 3.0)
        outliers = (z > threshold).any(axis=1)
        
        # Show count of outliers
        st.write(f"Number of Outliers Detected: {outliers.sum()}")
        
        # Provide options for handling outliers
        option = st.sidebar.selectbox("Handle Outliers", ["None", "Drop Outliers", "Replace with Mean"])
        
        if option == "Drop Outliers":
            st.write("Dropping rows with outliers...")
            data = data[~outliers]
        elif option == "Replace with Mean":
            st.write("Replacing outliers with column mean...")
            for col in data.select_dtypes(include=[np.number]).columns:
                data.loc[outliers, col] = data[col].mean()
        
    return data
