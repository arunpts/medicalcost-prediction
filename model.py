import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

data = pd.read_csv("insurance.csv")

columns = ['sex','smoker']
for column in columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
data['region'] = data['region'].apply({'southwest':1, 'southeast':2, 'northwest':3, 'northeast':4}.get)

x = data.drop(['charges'],axis=1)
y = data["charges"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state = 0)

rfr=RandomForestRegressor(n_estimators = 10,random_state= 0)
rfr.fit(x_train, y_train)



pickle.dump(rfr,open('model.pkl','wb'))
