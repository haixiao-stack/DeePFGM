import numpy as np
import torch
import matplotlib.pyplot as plt
from RES import ResNet,BasicBlock
from scipy import stats
from scipy.special import inv_boxcox
name="omegac"
worktype="data-4D"
networktype="networks-4D"
def process_raw(x):
    xmax=np.load("../../"+worktype+"/process_params/xmax.npy")
    xmin=np.load("../../"+worktype+"/process_params/xmin.npy")
    for i in range(4):
        x[:,i]=(x[:,i]-xmin[i])/(xmax[i]-xmin[i])
    return x
ytrue=np.load('../../'+worktype+'/raw_data/Ydata_all.npy')
x=np.load('../../'+worktype+'/raw_data/Xdata_all.npy')
# x=np.array([[0.1561,0],[0.1561,0],[0.1561,0],[0.1561,0],[0.1561,0]])
# print(np.shape(x))
x=process_raw(x)
# print(np.shape(x))
x = torch.tensor(x,dtype=torch.float)
# print(np.shape(x))
x=x.to("cuda")
model = ResNet(BasicBlock,9,300,4,3)
params = torch.load("../../"+networktype+"/model_res_"+name+".pth") # 加载参数,+name+
model.load_state_dict(params) # 应用到网络结构中
model=model.to("cuda")
with torch.no_grad():   
    predictions=model(x)
predictions=predictions.to("cpu")
predictions=predictions.data.numpy()
phimax=np.load("../../"+worktype+"/process_params/phimax_"+name+".npy")
phimin=np.load("../../"+worktype+"/process_params/phimin_"+name+".npy")
lambdas=np.load("../../"+worktype+"/process_params/lambdas_"+name+".npy")
constants=np.load("../../"+worktype+"/process_params/constants_"+name+".npy")
# print(np.shape(predictions))
# diff=0
# f, ax = plt.subplots(1,2,figsize=(6,3))
# plt.supxlabel("Flare-tbl")
# plt.supylabel("Flare-dnn")
#设置主标题
#设置子标题
names=["$\omega_{c}$","$c\omega_{c}$","$z\omega_{c}$","$C_{p}$","$m_{\omega}$","$H_{f_{0}}$","$T$","$ \mu$","$Y_{H_{2}O}$","$Y_{CO}$","$Y_{CO_{2}}$"]
ind=0
diff=0
predictions[:,ind-diff]=predictions[:,ind-diff]*(phimax[ind]-phimin[ind])+phimin[ind]
predictions[:,ind-diff]=inv_boxcox(predictions[:,ind-diff], lambdas[ind])-constants[ind]
phi_mean=np.mean(ytrue[:,ind])
# print(predictions[:,ind])
# print(ytrue[:,ind])
# print(np.mean((predictions[:,ind]-ytrue[:,ind])**2))
# print(np.mean((phi_mean-ytrue[:,ind])**2))
# errors=predictions[:,ind-diff]-ytrue[:,ind]
# print(predictions[:,ind-diff])
nan_positions = np.isnan(predictions[:,ind-diff])
# print(np.shape(np.isnan(predictions[:,ind-diff])))
# 仅保留非NaN值的位置
non_nan_positions = ~nan_positions
# 从原始数组中提取所有非NaN元素
cal1=predictions[non_nan_positions,ind-diff]
cal2=ytrue[non_nan_positions,ind]
print(np.shape(predictions[:,ind-diff])[0]-np.shape((cal1))[0])
# print(cal1)
# print(np.shape(cal2))
predictions[:,ind-diff] = np.nan_to_num(predictions[:,ind-diff], nan=0)#predictions[:,ind-diff]
print("R",1-np.mean((cal1-cal2)**2)/np.mean((phi_mean-ytrue[:,ind])**2))
# plt.subplot(1,2,1)
plt.scatter(cal1,cal2)#ytrue[:,ind],predictions[:,ind-diff]
# plt.title(names[ind])
# plt.subplot(1,2,2)
# usemax=np.max(cal1)
# usemin=np.min(cal1)
# list1=[]
# list1=np.argwhere(cal1<usemin+(usemax-usemin)*0.05)
# # list1=np.argwhere(cal1>usemin+(usemax-usemin)*0.2)
# xaxis=np.delete(cal1, list1, 0)
# yaxis=np.delete(cal2-cal1, list1, 0)
# plt.scatter(xaxis,yaxis)#yaxis/xaxis
# plt.title(names[ind])
plt.show()
# np.save("./data/predictions/yh2o",predictions[:,ind-diff])