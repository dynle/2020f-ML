import pandas as pd
import numpy as np
data=pd.read_csv("death.csv")
y=data["uk"][100:299]
x=np.arange(100,299)

from scipy.optimize import curve_fit
def func(x,a,b,c,d,e):
 return a*pow(x,4)+b*pow(x,3)+c*pow(x,2)+d*pow(x,1)+e
param=curve_fit(func,x,y)
[a,b,c,d,e]=param[0]
print(a,b,c,d,e)

import matplotlib.pyplot as plt
plt.plot(x,y)
plt.plot(x,func(x,a,b,c,d,e))

print(int(func(270,a,b,c,d,e)))
print(int(func(300,a,b,c,d,e)))

x_new = np.linspace(100,310) #extend the dataset
#x_new = np.arange(100,310) this can be possible
plt.plot(x_new, func(x_new,a,b,c,d,e)) #extend the curve_fit
plt.plot(300,int(func(300,a,b,c,d,e)),'ro')
plt.plot(310,int(func(310,a,b,c,d,e)),'ro')

plt.show()
