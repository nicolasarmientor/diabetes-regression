import pandas as pd
from sklearn.datasets import load_diabetes

def diabetes_data_extraction() -> pd.DataFrame:
    data = load_diabetes()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y 