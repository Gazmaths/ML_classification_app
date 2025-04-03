import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(best_models, X_test, y_test):
    results = {}
    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred, output_dict=True)
        }
    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])[0]
    return best_model, best_models[best_model], results

def feature_importance(model, X_test):
    importances = model.feature_importances_
    feature_names = X_test.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Plot using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    ax.invert_yaxis()
    ax.set_title('Feature Importances (Random Forest)')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    plt.tight_layout()

    # Display plot in Streamlit
    st.pyplot(fig)
    
    return feature_importance_df
