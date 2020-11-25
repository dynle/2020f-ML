import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

data=pd.read_csv('euro.csv')
x=np.arange(120,240).reshape(120, -1)
y=data['death'][120:240]
clf=RandomForestRegressor(n_estimators=82, min_samples_split=2)
clf.fit(x,y)
print(clf.score(x,y))
print(clf.feature_importances_)

p=clf.predict(x)

X=[300]
print("Dec.23=",int(clf.predict([X]))) #it does not use a trend

plt.plot(x,y,'--b')
plt.plot(x,p,'-r')
plt.legend(('real','randf'))
plt.show()
