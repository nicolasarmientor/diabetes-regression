import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from data_extraction import diabetes_data_extraction

def train_and_score() -> pd.DataFrame:
    X, y = diabetes_data_extraction()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=0.01, max_iter=1000),
        "Lasso": Lasso(alpha=0.01, max_iter=1000),
        "Huber": HuberRegressor(alpha=0.01, max_iter=1000)
    }

    rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        rows.append({
            "Model": name, 
            "RMSE": rmse, 
            "R2": r2
            })

    results = pd.DataFrame(rows).sort_values("R2", ascending=False)

    return results