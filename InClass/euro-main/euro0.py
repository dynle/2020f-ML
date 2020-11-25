import pandas as pd
import numpy as np
data=pd.read_csv("euro.csv")
y=data["death"]
x=np.arange(0,240)
import matplotlib.pyplot as plt
plt.plot(x,y)
plt.show()
