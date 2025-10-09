import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib

def evaluate():
    X_test = pd.read_csv("X_test.csv")
    y_test = pd.read_csv("y_test.csv")


    model = joblib.load("models/model.pkl")
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test,predictions)
    print(f"model evaluate completed MSE : {mse}")

if __name__ == "__main__":
    evaluate()