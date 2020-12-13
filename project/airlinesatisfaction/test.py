# get data from https://www.kaggle.com/teejmahal20/airline-passenger-satisfaction?select=train.csv

# from sklearn.preprocessing import StandardScaler

import pandas as pd
#get DataFrame
train=pd.read_csv("train.csv")
train=train.drop(['Unnamed: 0','id'],axis=1)

test=pd.read_csv("test.csv")
test=test.drop(['Unnamed: 0','id'],axis=1)



#check the data
#train.info(), train.describe(), train.head(), etc

#how to find empty data spaces
# train.isnull().sum()

#fill empty data in arrival delay in minutes with mean
# for dt in data_list:
#   mean = round(dt['Arrival Delay in Minutes'].mean())
#   dt['Arrival Delay in Minutes'].fillna(mean,inplace=True)
#   dt.fillna("",inplace=True)

import numpy as np
from sklearn.impute import SimpleImputer

# list the two
data_list=[train,test]
for dt in data_list:
  imp = SimpleImputer(missing_values=np.nan, strategy='mean')
  imp = imp.fit(dt[['Arrival Delay in Minutes']])
  dt['Arrival Delay in Minutes'] = imp.transform(dt[['Arrival Delay in Minutes']])

from sklearn.preprocessing import LabelEncoder
# str -> int

le = LabelEncoder()
for dt in data_list:
  for i in dt.columns.values.tolist():
    if type(dt[i][0])==str:
      dt[i] = le.fit_transform(dt[i])
    else:
      pass

import matplotlib.pyplot as plt 
# # See the data whether there are obvious outliers or not
# plt.figure()
# xx=np.linspace(0,len(train),len(train))
# plt.scatter(xx, train['Arrival Delay in Minutes'])
# plt.xlabel('passengers')
# plt.ylabel('Arrival Delay in Minutes')
# plt.show()


#  Get IQR
# q1= np.percentile(train['Arrival Delay in Minutes'],25)
# q3= np.percentile(train['Arrival Delay in Minutes'],75)
# iqr = q3 - q1
# lower_bound = q1 - (iqr*1.5)
# upper_bound = q3 + (iqr*1.5)
# print(q1)
# print(q3)


# outliers detection and removal (to increase the precision of ensemble model)
#clean up the data by removing the outliers

# satisfaction bar chart
# plt.figure(figsize = (8,5))
# train.satisfaction.value_counts(normalize = True).plot(kind='bar', color= ['darkorange','steelblue'],rot=0)
# plt.title('Satisfaction Indicator satisfied(0) and neutral or dissatisfied(1) in the Dataset')
# plt.show()

###########################################################

# #dataset
y_train=train["satisfaction"]
x_train=train.drop(["satisfaction"],axis=1)
y_test=test["satisfaction"]
x_test=test.drop(["satisfaction"],axis=1)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, ExtraTreesRegressor,ExtraTreesClassifier, AdaBoostRegressor,AdaBoostClassifier, BaggingRegressor, StackingRegressor, VotingRegressor

# training
clf=RandomForestClassifier(n_estimators=100,max_depth=None,min_samples_split=2,random_state=8)
clf.fit(x_train,y_train)
acc_rfc= (clf.score(x_test,y_test),'RandomForestClassifier')
print(f"accuracy RandomForestClassifier :{acc_rfc[0]}")

clf2=RandomForestRegressor(n_estimators=100,max_depth=None,min_samples_split=2,random_state=8)
clf2.fit(x_train,y_train)
acc_rfr= (clf2.score(x_test,y_test),'RandomForestRegressor')
print(f"accuracy RandomForestRegressor :{acc_rfr[0]}")

# clf3=ExtraTreesClassifier(n_estimators=100,max_depth=None,min_samples_split=2,random_state=8)
# clf3.fit(x_train,y_train)
# acc_etr = (clf3.score(x_test,y_test),'ExtraTreeClassifier')
# print(f"accuracy ExtraTreesClassifier :{acc_etr[0]}")

# clf4=ExtraTreesRegressor(n_estimators=100,max_depth=None,min_samples_split=2,random_state=8)
# clf4.fit(x_train,y_train)
# acc_etr = (clf4.score(x_test,y_test),'ExtraTreeRegressor')
# print(f"accuracy ExtraTreesRegressor :{acc_etr[0]}")

# clf5=AdaBoostClassifier(n_estimators=100,random_state=8)
# clf5.fit(x_train,y_train)
# acc_abc = (clf5.score(x_test,y_test),'AdaBoostClassifier')
# print(f"accuracy AdaBoostClassifier:{clf5.score(x_test,y_test)}")

# clf6=AdaBoostRegressor(n_estimators=100,random_state=8)
# clf6.fit(x_train,y_train)
# acc_abr = (clf6.score(x_test,y_test),'AdaBoostRegressor')
# print(f"accuracy AdaBoostRegressor:{clf5.score(x_test,y_test)}")

# #compare the accuracy rates in a chart
all_models=[acc_rfc,acc_rfr]
all_models=sorted(all_models,key=lambda x:x[0])

accuracy_values=[x[0] for x in all_models]
models=[x[1] for x in all_models]

plt.barh(models,accuracy_values,align='center',height=0.5, color='steelblue')
for i, v in enumerate(accuracy_values):
    plt.text(v,i-.05,str(round(v,3)),color='black',fontweight='bold')
plt.ylabel('Models')
plt.xlabel('Accuracy')
plt.title('Accuracy values',fontweight='bold')
plt.show()


# print feature importances in order
dic=dict(zip(x_train.columns,clf.feature_importances_))
for item in sorted(dic.items(), key=lambda x: x[1], reverse=True):
    print(item[0],round(item[1],4))

# make a horizontal bar chart
importances = pd.Series(data=clf.feature_importances_,index= x_train.columns)
importances_sorted = importances.sort_values()
plt.figure(figsize=(9,9))
for i, v in enumerate(importances_sorted):
    plt.text(v,i-.2,str(round(v,3)),color='black',fontweight='bold')
importances_sorted.plot(kind='barh')
plt.title('Features Importances')
plt.show()
