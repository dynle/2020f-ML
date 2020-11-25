'''
# ASSIGNMENT:
In order to use x=data[['date','temp','street']], you must modify the values of 'date'.
HINTs:
use str.replace function: dt['date']=dt['date'].str.replace...
use datetime: from datetime import datetime
use datetime(...).strftime('%w'): 2012/8/1 -> 2012,8,1 -> 3 (wednesday)
create test.csv where dt['date'] includes integers.
'''

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from datetime import datetime

data=pd.read_csv("ice.csv")
data["date"]=data["date"].str.replace("/",",")
for i in data["date"]:
    print(i)
    data["date"]=data["date"].replace(
        i,f"{datetime(int(i[:4]),int(i[5]),int(i[7:])).strftime('%w')}")
# print(data["date"])

data.to_csv("test.csv",mode="w",index=False)

'''
# read ice.csv file
data=pd.read_csv('ice.csv')
# y=f(x)=f(temp,street) where y is ice sales which we would like to predict
# x is temp and street
x=data[['temp','street']]
y=data['ice']
# RandomForestRegressor: n_estimators is no. of trees
# min_samples_split specifies the minimum number of samples required 
# to split an internal leaf node.
clf=RandomForestRegressor(n_estimators=50, min_samples_split=2)
# Machine Learning
clf.fit(x,y)
# accuracy score
print(clf.score(x,y))
# feature_importances_
print(clf.feature_importances_)
# predict p=f(x)
p=clf.predict(x)
t=np.arange(0.0,31.0)
plt.plot(t,data['ice'],'--b')
plt.plot(t,p,'-b')
plt.legend(('real','randomF'))
plt.show()
'''