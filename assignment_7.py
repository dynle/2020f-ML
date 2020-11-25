# sort dd based on 'ice' value.

import pandas as pd

d=pd.read_csv("ice.csv")
dd=d.drop(["date","temp"],axis=1)
print(dd.sort_values("ice"))