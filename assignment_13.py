# ASSIGNMENT: random forest using ice.csv
# build regression program and classification programs using ice.csv respectively.
# Use train_test_split function 
# HINT:
# from sklearn.model_selection import train_test_split
# test_size=0.2 indicates testing 20% and training 80%.
# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=54,shuffle=True)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor #regression program
from sklearn.ensemble import RandomForestClassifier #classification program
import matplotlib.pyplot as plt

#splitting data process
dt=pd.read_csv("ice.csv")
x=dt[['temp','street']]

# dt=pd.read_csv("test.csv")
# x=dt[['date','temp','street']]

y=dt['ice']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=54,shuffle=True) #more test_size, more accuracy

#training Regression
# clf=RandomForestRegressor(n_estimators=100,min_samples_split=2)

#training Classification
clf=RandomForestClassifier(n_estimators=100,min_samples_split=2)

clf.fit(x_train,y_train)

#data info & input new data to the trained program
print(clf.score(x,y))
print(clf.feature_importances_)

p=clf.predict(x_test)
print(p)
print(y_test)
t=np.arange(0.0,float(len(y_test)))

plt.plot(t,y_test,'--r') #original data
plt.plot(t,p,'-b') #predicted data
plt.legend(('real','randomF'))
plt.show()


