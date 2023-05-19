from scipy import stats
from scipy.special import inv_boxcox
import numpy as np
Xdata=np.load('./data/raw_data/Xdata_T.npy')
Ydata=np.load('./data/raw_data/Ydata_T.npy')
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
for i in range(8):
    x[:,i]=(x[:,i]-xmin[i])/(xmax[i]-xmin[i])
np.save("./data/train_data/Xdata_train_T",x)
np.save("./data/train_data/Ydata_train_T",y)
np.save("./data/process_params/phimax_T",phimax)
np.save("./data/process_params/phimin_T",phimin)
# np.save("./data/process_params/xmax_try",xmax)
# np.save("./data/process_params/xmin_try",xmin)
np.save('./data/process_params/constants_T',constants)
np.save('./data/process_params/lambdas_T',lambdas)
