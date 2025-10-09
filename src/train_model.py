import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


def train():
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Load training data
    x_train = pd.read_csv('x_train.csv')
    y_train = pd.read_csv('y_train.csv')

    # Train model
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Save trained model
    joblib.dump(model, 'models/model.pkl')
    print("Model training completed successfully.")


if __name__ == "_main_":
    train()