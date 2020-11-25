import pandas as pd
import numpy as np
df=pd.read_excel("deaths_in_europe.csv",sheet_name='Data')
n=len(df["Unnamed: 2"])
y=df["Unnamed: 2"][4:n]
y=y.replace("-",0)
x=np.arange(4,n)

trendpoly=np.poly1d(np.polyfit(x,y,10))

xp=np.linspace(4,n)

import matplotlib.pyplot as plt
plt.plot(x,y,'-k',label='data')
plt.plot(xp,trendpoly(xp),'--r',label='trend')
plt.legend()
plt.show()