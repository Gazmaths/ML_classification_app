import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def explore_data(data, target_column):
    if st.sidebar.checkbox("Correlation with Target"):
        corr = data.corr()[target_column].sort_values(ascending=False)
        st.write(corr)
    
    if st.sidebar.checkbox("Pairplot"):
        fig = sns.pairplot(data)  # This creates a figure
        st.pyplot(fig.figure)     # Use the figure from the pairplot object
