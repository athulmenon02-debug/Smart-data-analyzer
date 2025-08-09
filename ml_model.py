import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

def train_and_predict(df: pd.DataFrame, target: str, test_size: float = 0.2):
    data = df.dropna(subset=[target]).copy()
    if data[target].dtype.kind in 'if':
        task = "Regression"
        X = data.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
        y = data[target]
        model = LinearRegression()
    else:
        task = "Classification"
        X = pd.get_dummies(data.drop(columns=[target]))
        y = data[target]
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    if task == "Regression":
        mse = mean_squared_error(y_test, preds)
        result = {"name": "MSE", "value": mse}
    else:
        acc = accuracy_score(y_test, preds)
        result = {"name": "Accuracy", "value": acc}

    info = {"task_type": task, "model_name": type(model).__name__}
    return result, info
