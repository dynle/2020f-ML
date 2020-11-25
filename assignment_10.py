import matplotlib.pyplot as plt
import pandas as pd

data=pd.read_csv("./python-novice-master/f.csv")
plt.plot(data["f"],data["num"],"bo")
fig=plt.figure(1)
fig.set_size_inches(5,5)
fig.show()