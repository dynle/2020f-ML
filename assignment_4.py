# read ice.csv file and find the largest value in ice.
# show street value when ice is the max value.

import pandas as pd

#read a csv file
d=pd.read_csv("ice.csv")

#find the largest value in ice
m=d["ice"].max()
print(m)

#street value when ice is the max value
m_idx=d["ice"].idxmax()
print(d["street"][m_idx])