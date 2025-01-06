import pandas as pd
from sklearn.metrics import classification_report
import shap

def evaluate_model(best_models, X_test, y_test):
    results = {}
    for name, model in best_models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "report": classification_report(y_test, y_pred)
        }
    best_model = max(results.items(), key=lambda x: x[1]["accuracy"])[0]
    return best_model, best_models[best_model], results

def feature_importance(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)
