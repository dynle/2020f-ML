import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv("diabetes_data_upload.csv")
# x=df[['Age','Gender','Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia','Genital thrush','visual blurring','Itching','Irritability','delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity']]
# y=df['class']

# str -> int
le = LabelEncoder()
for i in df.columns.values.tolist():
 if (i=='Age'):
  pass
 else:
  df[i] = le.fit_transform(df[i])

#dataset
target = df['class']
data = df.drop(['class'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=54,shuffle=True)

#create a csv in the edited version
yX = target
yX = pd.concat([yX,data],axis=1)
yX.to_csv("diabetes_new.csv",encoding="utf-8")

#training
clf = RandomForestClassifier(n_estimators=382,max_depth=None,min_samples_split=2,random_state=8)
clf.fit(x_train,y_train)

print(clf.score(x_test,y_test),"\n")

#feature importances in order
dic = dict(zip(data.columns,clf.feature_importances_))
for item in sorted(dic.items(),key=lambda x: x[1], reverse=True):
    print(item[0],round(item[1],4))

#new data
Xnew=[[50,'Male','No','No','No','No','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes','Yes']]
df_new=pd.DataFrame(Xnew,columns=['Age','Gender','Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia','Genital thrush','visual blurring','Itching','Irritability','delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity'])

#change str to int, should be better way to do this. ex LabelEncoder
for i in df_new.columns.values.tolist():
    if (i=='Age'):
        pass
    elif (i=='Gender'):
        if (df_new[i].item()=='Male'): df_new[i]=1
        else: df_new[i]=0
    else:
        if (df_new[i].item()=='Yes'): df_new[i]=1
        else: df_new[i]=0
        
print(df_new)
pred_new=clf.predict(df_new)
print(pred_new)
# pred=clf.predict(x_test)
# print(pred)
# df2=pd.DataFrame({'Actual':y_test,'Predicted':pred})
# print(df2)

