#subtract each element in two lists

#1)
from operator import sub
a=[0,0,1,0]
b=[1,0,1,0]
c=list(map(sub,a,b))
print(c)

#2)
import numpy as np
c=np.array(a)
d=np.array(b)
e=c-d
print(e)