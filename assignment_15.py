# build a program using dataset of machine-learning-in-medicine/pima-indians-diabetes.csv 
# ASSIGNMENT: binary classification with SMOTE
# HINT:
# use SMOTE for imbalanced data in pima-indians-diabetes.csv.

import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

#get DataFrame
pima=pd.read_csv("pima-indians-diabetes.csv",encoding="utf-8",header=None)
pima.columns=['pregnant','plasmaGlucose','bloodP','skinThick','serumInsulin','weight','pedigree','age','diabetes']

y = pima["diabetes"]
X = pima.drop(["diabetes"],axis=1)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=54,shuffle=True)

#SMOTE over_sampling
smt = SMOTE(random_state=90)
X_train,y_train = smt.fit_resample(X_train,y_train)

#training
clf=RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_split=2,random_state=8)
clf.fit(X_train,y_train)

print(clf.score(X_test,y_test))

#print feature importances in order
dic=dict(zip(X.columns,clf.feature_importances_))
for item in sorted(dic.items(), key=lambda x: x[1], reverse=True):
    print(item[0],round(item[1],4))