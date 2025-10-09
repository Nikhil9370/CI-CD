import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess():

    data=pd.DataFrame({
        "x": range(10) , 
        "y": [2*i+1 for i in range (10)]
    })

    X = data[['x']]
    y = data['y']

    X_train , X_test , y_train , y_test = train_test_split(X ,y , test_size=0.2 , random_state=42)

    X_train.to_csv("X_train.csv",Index=False)
    X_test.to_csv("X_test.csv",Index=False)
    y_train.to_csv("y_train.csv",Index=False)
    y_test.to_csv("y_test.csv",Index=False)

    print("Data Preprocessing Completed")

if __name__ == "__main__":
    preprocess()