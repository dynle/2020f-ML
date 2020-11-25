import pandas as pd
import numpy as np
data=pd.read_csv("death.csv")
y=data["us"][100:299]
x=np.arange(100,299)

from scipy.optimize import curve_fit
def func(x,a,b,c,d):
 return a*x*x*x+b*x*x+c*x+d
param=curve_fit(func,x,y)
[a,b,c,d]=param[0]
print(a,b,c,d)

import matplotlib.pyplot as plt
plt.plot(x,y)
plt.plot(x,func(x,a,b,c,d))

x_new = np.linspace(100,350) #extend the dataset 
plt.plot(x_new, func(x_new,a,b,c,d)) #extend the curve_fit
plt.plot(300,int(func(300,a,b,c,d)),'ro')
plt.plot(350,int(func(350,a,b,c,d)),'ro')

plt.show()