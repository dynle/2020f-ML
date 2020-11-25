# ASSIGNMENT: binary classification using random forest classification.
# read titanic folder and develope a binary classification program
# https://github.com/ytakefuji/titanic
# use train_test_split function.
# show what are importances in the features.
# You must understand preprocessing and train_test_split.

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

titanic = pd.read_csv("titanic.csv")
titanic = titanic.drop(['name','row.names'],axis=1) #delete unnecessary info from the data

#fill empty data in age with mean
mean = round(titanic['age'].mean(),2)
titanic['age'].fillna(mean,inplace=True)
titanic.fillna("",inplace=True)

# str -> int
le = LabelEncoder()
for i in titanic.columns.values.tolist():
 if (i=='age'):
  pass
 else:
  titanic[i] = le.fit_transform(titanic[i])

#dataset
titanic_target = titanic['survived']
titanic_data = titanic.drop(['survived'],axis=1)
X_train,X_test,y_train,y_test = train_test_split(titanic_data,titanic_target,test_size=0.2,random_state=54,shuffle=True)

#create a csv in the edited version
yX = titanic_target
yX = pd.concat([yX,titanic_data],axis=1)
yX.to_csv("titanic_new.csv",encoding="utf-8")

#training
clf = RandomForestClassifier(n_estimators=382,max_depth=None,min_samples_split=2,random_state=8)
clf.fit(X_train,y_train)

print(clf.score(X_test,y_test),"\n")

#feature importances in order
dic = dict(zip(titanic_data.columns,clf.feature_importances_))
for item in sorted(dic.items(),key=lambda x: x[1], reverse=True):
    print(item[0],round(item[1],4))