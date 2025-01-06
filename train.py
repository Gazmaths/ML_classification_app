from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

def train_models(data, target):
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(),
        "Logistic Regression": LogisticRegression(),
        "SVC": SVC()
    }

    params = {
        "Random Forest": {"n_estimators": [10, 50, 100]},
        "Logistic Regression": {"C": [0.01, 0.1, 1]},
        "XGBoost": {"learning_rate": [0.01, 0.1, 0.2]},
        "SVC": {"C": [0.1, 1, 10]}
    }

    best_models = {}
    for name, model in models.items():
        clf = GridSearchCV(model, params[name], cv=5)
        clf.fit(X_train, y_train)
        best_models[name] = clf.best_estimator_

    return best_models, X_test, y_test
