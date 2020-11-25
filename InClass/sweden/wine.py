# basic structrue on Takefuji github

import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,VotingClassifier,RandomForestClassifier,RandomForestRegressor,ExtraTreesRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data=pd.read_csv('red.csv')
x=data[['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol']]
y=data['quality']

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=54,shuffle=True)

# clf2=ExtraTreesClassifier(n_estimators=82, max_depth=None,min_samples_split=2, random_state=0)
# clf3=RandomForestClassifier(random_state=0,n_estimators=250, min_samples_split=2)
clf4=RandomForestRegressor(random_state=0,n_estimators=250,min_samples_split=2)
# clf = VotingClassifier(estimators=[ ('et', clf2),('rf',clf3)], voting='soft',weights=[4,1]).fit(X_train,y_train)

# clf2.fit(X_train,y_train)
# clf3.fit(X_train,y_train)
clf4.fit(X_train,y_train)

# print(clf.score(X_test,y_test))
# print(clf2.score(X_test,y_test))
# print(clf3.score(X_test,y_test))
print(clf4.score(X_test,y_test))

dic=dict(zip(x.columns,clf4.feature_importances_))
for item in sorted(dic.items(), key=lambda x: x[1], reverse=True):
    print(item[0],round(item[1],4))

# p=clf3.predict(X_test) #predict based on randomforestclassifer
# t=np.arange(0.0,float(len(y_test)))

# plt.plot(t,y_test,'--r') #original data
# plt.plot(t,p,'-b') #predicted data
# plt.legend(('real','randomF'))
# plt.show()