import pandas as pd
import numpy as np
data=pd.read_csv("euro.csv")
y=data["death"][120:240]
x=np.arange(120,240)

from scipy.optimize import curve_fit
def func(x,a,b,c,d):
 return a*x*x*x+b*x*x+c*x+d
param=curve_fit(func,x,y)
[a,b,c,d]=param[0]
print(a,b,c,d)

import matplotlib.pyplot as plt
plt.plot(x,y)
plt.plot(x,func(x,a,b,c,d))

print("Nov. 23 deaths",int(func(270,a,b,c,d)))
print("Dec. 23 deaths",int(func(300,a,b,c,d)))

x_new = np.linspace(120,300) #extend the dataset 
plt.plot(x_new, func(x_new,a,b,c,d)) #extend the curve_fit
plt.plot(270,int(func(270,a,b,c,d)),'ro')
plt.plot(300,int(func(300,a,b,c,d)),'ro')

plt.show()
