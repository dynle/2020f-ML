# show ice value when street is the max value.

import pandas as pd

data=pd.read_csv("ice.csv")

m_idx=data["street"].idxmax()
print(data["ice"][m_idx])