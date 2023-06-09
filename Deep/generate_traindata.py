from scipy import stats
from scipy.special import inv_boxcox
import numpy as np
worktype="data-4D"
name="all"
Xdata=np.load("../../"+worktype+"/raw_data/Xdata_"+name+".npy")
Ydata=np.load("../../"+worktype+"/raw_data/Ydata_"+name+".npy")
print(np.shape(Ydata))
length=len(Ydata[0])
lambdas=np.zeros(length)
constants=np.zeros(length)
for i in range(length):
    ymin=np.min(Ydata[:,i])
    if(ymin<=0):
        constant=-ymin+1e-4 
    else:
        constant=0
    constants[i]=constant
    Ydata[:,i], lambda_ = stats.boxcox(Ydata[:,i]+constant)
    lambdas[i]=lambda_
label_texts=["$\omega_{c}$","$c\omega_{c}$","$z\omega_{c}$","$C_{p}$","$m_{\omega}$","$H_{f_{0}}$","$T$","$ \mu$","$Y_{H_{2}O}$","$Y_{CO}$","$Y_{CO_{2}}$"]
x=Xdata
y=Ydata
xmin=np.min(x,axis=0)
xmax=np.max(x,axis=0)
phimin=np.min(y,axis=0)
phimax=np.max(y,axis=0)
for i in range(len(label_texts)):
    y[:,i]=(y[:,i]-phimin[i])/(phimax[i]-phimin[i])
for i in range(2):
    x[:,i]=(x[:,i]-xmin[i])/(xmax[i]-xmin[i])
print(np.shape(y))
np.save("../../"+worktype+"/train_data/Xdata_train_"+name+".npy",x)
np.save("../../"+worktype+"/train_data/Ydata_train_"+name+".npy",y)
np.save("../../"+worktype+"/process_params/phimax_"+name+".npy",phimax)
np.save("../../"+worktype+"/process_params/phimin_"+name+".npy",phimin)
np.save("../../"+worktype+"/process_params/constants_"+name+".npy",constants)
np.save("../../"+worktype+"/process_params/lambdas_"+name+".npy",lambdas)
