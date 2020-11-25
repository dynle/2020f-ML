import pandas as pd
import numpy as np
data=pd.read_excel(open("world.xlsx","rb"))
y=data["day"][100:287]
x=np.arange(100,287)

from scipy.optimize import curve_fit

def func(x,a,b,c,d):
 return a*x*x*x+b*x*x+c*x+d

param=curve_fit(func,x,y)
[a,b,c,d]=param[0]
print(a,b,c,d)

import matplotlib.pyplot as plt
plt.plot(x,y) #real
plt.plot(x,func(x,a,b,c,d)) #curve_fit

x_new = np.linspace(100,320) #extend the dataset 
plt.plot(x_new, func(x_new,a,b,c,d)) #extend the curve_fit
plt.plot(300,int(func(300,a,b,c,d)),'ro')
plt.plot(320,int(func(320,a,b,c,d)),'ro')

plt.show()
