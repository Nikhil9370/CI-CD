import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib

def evaluate():
    x_test = pd.read_csv('x_test.csv')
    y_test = pd.read_csv('y_test.csv')

    model = joblib.load('models/model.pkl')
    pridictions = model.pridict(x_test)

    mse = mean_squared_error(y_test,pridictions)
    print(f"model evaluation complated MSE : {mse}")

    if __name__ == "_main_":
        evaluate()