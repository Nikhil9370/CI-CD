import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess():

    data=pd.DataFrame({
        'x':range(10),
        'y':[2*i+1 for i in range(10)]
    })
    x=data[['x']]
    y=data[['y']]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

    x_train.to_csv('x_train.csv',index=False)
    x_test.to_csv('x_test.csv',index=False)
    y_train.to_csv('y_train.csv',index=False)
    y_test.to_csv('y_test.csv',index=False)

    print("Data Preprocessing complated")

if __name__ =="main_":
    preprocess()