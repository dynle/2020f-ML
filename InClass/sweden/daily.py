import pandas as pd
import numpy as np
data=pd.read_csv("new_deaths.csv")
country = input("country name: ")

y=data[country][72:318]
x=np.arange(72,318)

trend=np.polyfit(x,y,10)
trendpoly=np.poly1d(trend)

xp=np.linspace(80,320)

import matplotlib.pyplot as plt
plt.plot(x,y,'-k',label="data")
plt.plot(xp,trendpoly(xp),'-b',label="trend")
plt.ylim(y.min()-1,y.max()+5)
plt.xlim(50,340)
plt.legend()
plt.show()

"""
import pandas as pd
import numpy as np
data=pd.read_csv("new_deaths.csv")
country=input("country name: ")
n=len(data[country])

day1=245
y1=data[country][n-day1:n]
x1=np.arange(n-day1,n)

day2=100
y2=data[country][n-day2:n]
x2=np.arange(n-day2,n)

def valid(x,y):
    validity = ~(np.isnan(x) | np.isnan(y))
    return validity

trendpoly1=np.poly1d(np.polyfit(x1[valid(x1,y1)],y1[valid(x1,y1)],10))
trendpoly2=np.poly1d(np.polyfit(x2[valid(x2,y2)],y2[valid(x2,y2)],10))

xp1=np.linspace(n-day1,n+7)
xp2=np.linspace(n-day2,n+7)

import matplotlib.pyplot as plt
plt.plot(x1,y1,'-k',label="daily deaths in Sweden")
plt.plot(xp1,trendpoly1(xp1),'-b',label="245 days from Nov.11")
plt.plot(xp2,trendpoly2(xp2),'--r',label="110 days from Nov.11")
plt.ylim(y1.min()-1,y1.max()+5)
plt.legend()
plt.show()
"""


