import pandas as pd
import numpy  as np
from sklearn.model_selection import train_test_split
data=np.array([])
x=np.zeros(4)
y=np.zeros(14)
num=80
columns=19
rows=21735
for i in range(1,num+1):#1,num+1
 str_i=str("%02d"%i)
#  data_i = np.fromfile("./data_new/unit"+str_i+"_h01.dat", dtype=np.float64, count=rows*columns, sep=" ", offset=0)
#  data_i=np.reshape(data_i,(rows,columns))
 data_i = np.loadtxt("../../data-4D/data-2.0/unit"+str_i+"_h01.dat", dtype=np.float64)
 x_i=data_i[:,0:4]
 y_i=data_i[:,5:19]
 x=np.vstack([x,x_i])
 y=np.vstack([y,y_i])
 print(i)
# print(i)
y=np.delete(y,[8,9,10],axis=1) 
x=x[1:]
y=y[1:]
# new_col = np.array(x[:,0]*(1/(x[:,2]+1e-4)-1))
# x = np.c_[x, new_col]
# new_col = np.array((1-x[:,0])*(1/(x[:,2]+1e-4)-1))
# x = np.c_[x, new_col]
# new_col = np.array(x[:,1]*(1/(x[:,3]+1e-4)-1))
# x = np.c_[x, new_col]
# new_col = np.array((1-x[:,1])*(1/(x[:,3]+1e-4)-1))
# x = np.c_[x, new_col]
np.save('../../data-4D/raw_data/Xdata_all',x)
np.save('../../data-4D/raw_data/Ydata_all',y)