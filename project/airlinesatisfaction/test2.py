# get data from https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction?select=train.csv

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 

#get DataFrame
train=pd.read_csv("train.csv",encoding="shift-jis")
train=train.drop(['Unnamed: 0','id'],axis=1) #drop unnecessary info
test=pd.read_csv("test.csv",encoding="shift-jis")
test=test.drop(['Unnamed: 0','id'],axis=1) #drop unnecessary info

data_list=[train,test]

#fill empty data in 'arrival delay in minutes' with mean // how to find the NaN place
for dt in data_list:
  mean = round(dt['Arrival Delay in Minutes'].mean())
  dt['Arrival Delay in Minutes'].fillna(mean,inplace=True)
  dt.fillna("",inplace=True)

#str -> int
le = LabelEncoder()
for dt in data_list:
  for i in dt.columns.values.tolist():
    if i=='Gender' or i=='Customer Type' or i=='Type of Travel' or i=='Class' or i=='satisfaction':
      dt[i] = le.fit_transform(dt[i])
    else:
      pass

#dataset
y_train=train["satisfaction"]
x_train=train.drop(["satisfaction"],axis=1)
y_test=test["satisfaction"]
x_test=test.drop(["satisfaction"],axis=1)

#training
clf=RandomForestRegressor(n_estimators=100,max_depth=None,min_samples_split=2,random_state=8)
clf.fit(x_train,y_train)

print(f"randomforestregressor accuracy :{clf.score(x_test,y_test)}\n")

#print feature importances in order
dic=dict(zip(x_train.columns,clf.feature_importances_))
for item in sorted(dic.items(), key=lambda x: x[1], reverse=True):
    print(item[0],round(item[1],4))