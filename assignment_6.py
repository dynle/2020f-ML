# show max value in ice and street.
# show index of max value in ice and street.

import pandas as pd

data=pd.read_csv("ice.csv")
data2=data.drop(["date","temp"],axis=1)
print(data2["ice"].max())
print(data2["street"].max())
print(data["ice"].idxmax())
print(data["street"].idxmax())