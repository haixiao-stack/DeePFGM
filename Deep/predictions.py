import numpy as np
import torch
import matplotlib.pyplot as plt
from RES import ResNet,BasicBlock
from scipy import stats
from scipy.special import inv_boxcox
def process_raw(x):
    xmax=np.load("./data/process_params/xmax.npy")
    xmin=np.load("./data/process_params/xmin.npy")
    for i in range(8):
        x[:,i]=(x[:,i]-xmin[i])/(xmax[i]-xmin[i])
    return x
ytrue=np.load('./data/raw_data/Ydata.npy')
x=np.load('./data/raw_data/Xdata.npy')
x=process_raw(x)
x = torch.tensor(x,dtype=torch.float)
x=x.to("cuda")
model = ResNet(BasicBlock,6,300,8,1)
params = torch.load("./networks/model_gpu_400_cp.pth") # 加载参数
model.load_state_dict(params) # 应用到网络结构中
model=model.to("cuda")
with torch.no_grad():   
    predictions=model(x)
predictions=predictions.to("cpu")
predictions=predictions.data.numpy()
phimax=np.load("./data/process_params/phimax.npy")
phimin=np.load("./data/process_params/phimin.npy")
lambdas=np.load("./data/process_params/lambdas.npy")
constants=np.load("./data/process_params/constants.npy")
# print(np.shape(predictions))
# diff=0
f, ax = plt.subplots(1,2,figsize=(6,3))
# plt.supxlabel("Flare-tbl")
# plt.supylabel("Flare-dnn")
#设置主标题
#设置子标题
names=["$\omega_{c}$","$c\omega_{c}$","$z\omega_{c}$","$C_{p}$","$m_{\omega}$","$H_{f_{0}}$","$T$","$ \mu$","$Y_{H_{2}O}$","$Y_{CO}$","$Y_{CO_{2}}$"]
ind=5
diff=3
predictions[:,ind-diff]=predictions[:,ind-diff]*(phimax[ind]-phimin[ind])+phimin[ind]
predictions[:,ind-diff]=inv_boxcox(predictions[:,ind-diff], lambdas[ind])-constants[ind]
phi_mean=np.mean(ytrue[:,ind])
# print(predictions[:,ind])
# print(ytrue[:,ind])
# print(np.mean((predictions[:,ind]-ytrue[:,ind])**2))
# print(np.mean((phi_mean-ytrue[:,ind])**2))
# errors=predictions[:,ind-diff]-ytrue[:,ind]
predictions[:,ind-diff] = np.nan_to_num(predictions[:,ind-diff], nan=0)
print("R",1-np.mean((predictions[:,ind-diff]-ytrue[:,ind])**2)/np.mean((phi_mean-ytrue[:,ind])**2))
plt.subplot(1,2,1)
plt.scatter(ytrue[:,ind],predictions[:,ind-diff])
plt.title(names[ind])
plt.subplot(1,2,2)
usemax=np.max(ytrue[:,ind])
usemin=np.min(ytrue[:,ind])
# list1=np.argwhere(ytrue[:,ind]<usemin+(usemax-usemin)*0.2)
list1=np.argwhere(ytrue[:,ind]>usemin+(usemax-usemin)*0.2)
xaxis=np.delete(ytrue[:,ind], list1, 0)
yaxis=np.delete(predictions[:,ind-diff]-ytrue[:,ind], list1, 0)
plt.scatter(xaxis,yaxis/xaxis)
plt.title(names[ind])
plt.show()
# np.save("./data/predictions/yh2o",predictions[:,ind-diff])