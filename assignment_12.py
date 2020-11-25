# Machine Learning: Random Forest Classification (integer)
# ASSIGNMENT:
# use three parameters in x: x=date[['date','temp','street']]
# make a program to predict y=date['ice']: y=f(x)
# show feature_importances of ['date','temp','street']

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

data=pd.read_csv('test.csv')
x=data[['date','temp','street']] 
y=data['ice']
clf=RandomForestClassifier(n_estimators=82, min_samples_split=2)
clf.fit(x,y)

print(clf.score(x,y))
# print(clf.predict(x)) # predicted ice data
print(clf.feature_importances_)

p=clf.predict(x)
t=np.arange(0.0,31.0) # num of data

plt.plot(t,data['ice'],'--r')
plt.plot(t,p,'-b')
plt.legend(('real','randomF'))
plt.show()