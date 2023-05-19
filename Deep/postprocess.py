from RES import ResNet,BasicBlock
import numpy as np
import torch
import csv
import pandas as pd
phinum=6
for num in (1,2,3,15,30,45,60):
    data = np.genfromtxt('./data/postdata/'+str(num)+'.csv', delimiter=',')
    # print(data)
    # result = data * 2
    # np.savetxt('result.csv', result, delimiter=',')
    data=data[1:,[5,7,9,10]]
    x=data
    x_copy = np.copy(x)
    x_copy[:, [1, 2]] = x[:, [2, 1]]
    x=x_copy
    x[:,2]=x[:,2]/(x[:,0]*(1-x[:,0]))
    x[:,3]=x[:,3]/(x[:,1]*(1-x[:,1]))
    x = torch.tensor(x,dtype=torch.float)
    x_cuda=x.cuda()
    model = ResNet(BasicBlock,9,300,4,1)
    ResNet_cuda=  model.cuda()
    params = torch.load("./networks/model_gpu_1000_T.pth") # 加载参数
    ResNet_cuda.load_state_dict(params) # 应用到网络结构中
    if(torch.cuda.is_available()):
        with torch.no_grad():   
            predictions=ResNet_cuda(x_cuda)
    predictions=predictions.to("cpu")
    predictions=predictions.data.numpy()
    a,b=np.shape(predictions)
    predictions=np.reshape(predictions,(a,1))
    print(np.shape(predictions))
    phimax=np.load("./data/train_data/phimax.npy")
    phimin=np.load("./data/train_data/phimin.npy")
    for k in range(1):
        predictions[:,k]=predictions[:,k]*(phimax[phinum]-phimin[phinum])+phimin[phinum]
    dnnphi=np.reshape(predictions[:,0],(a,1))
    print(dnnphi)
    data2 = pd.DataFrame(dnnphi, columns=[str(phinum)+"-dnn"])
    # 读取 CSV 文件
    data1 = pd.read_csv("./data/postdata/"+str(num)+".csv")
    data1["T-dnn"] = data2[str(phinum)+"-dnn"]
    # 将更新后的数据写入 CSV 文件
    data1.to_csv("./data/postdata/output_T"+str(num)+".csv", index=False)