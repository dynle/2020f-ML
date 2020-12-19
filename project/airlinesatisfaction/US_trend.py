# https://www.statista.com/statistics/1102816/coronavirus-covid19-cases-number-us-americans-by-day/
import pandas as pd
import numpy as np
df=pd.read_excel("us_cases.xlsx",sheet_name='Data')
n=len(df["Unnamed: 2"])
y=df["Unnamed: 2"][4:n]
y=y.replace("-",0)
x=np.arange(4,n)

trendpoly=np.poly1d(np.polyfit(x,y,10))

xp=np.linspace(4,n)

import matplotlib.pyplot as plt
label=[["January 20","Febuary","March","April","May","June","July",
        "Auguest","September","Ocboter","November","December 13"]]
plt.figure(figsize=(8,8))
plt.plot(x,y,label='data')
plt.plot(x,trendpoly(x),'--r',label='trend',linewidth=2.5)
plt.xticks(np.arange(0,360,30),labels=label,rotation=45,fontsize=12)
plt.ylabel("Number of new cases",fontsize=15)
plt.legend(fontsize=20)
plt.show()

x_new=np.linspace(4,n+20)

plt.figure(figsize=(8,8))
plt.plot(x,y,label='data')
plt.plot(x_new,trendpoly(x_new),'--r',label='trend',linewidth=2.5)
plt.xticks(np.arange(0,360,30),labels=label,rotation=45,fontsize=12)
plt.ylabel("Number of new cases",fontsize=15)
plt.legend(fontsize=20)
plt.show()





