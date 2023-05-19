batch_size_train = 10240
lr_train = 1e-5 #学习率
weight_decay_train=1e-5 #梯度下降衰减系数
epochs_train = 5000 #训练迭代次数
blocknum=6
neuronum=300
inputnum=8
outputnum=11
import torch
import torch.utils.data as Data
from DNN import ResNet,BasicBlock
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
x=np.load('./data/train_data/Xdata_process_feature.npy')
y=np.load('./data/train_data/Ydata_process.npy')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.05)
model= ResNet(BasicBlock,blocknum,neuronum,inputnum,outputnum)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
# i=0
y_train_phi=y_train[:,:] 
y_test_phi=y_test[:,:]
ymean_error=np.load('./data/process_data/ymean_error.npy')
optimizer = torch.optim.Adam(model.parameters(),lr = lr_train,weight_decay=weight_decay_train)
loss_func = torch.nn.MSELoss()
# params = torch.load("model_gpu"+dirpart+"_omega_c.pth") # 加载参数
# model.load_state_dict(params) # 应用到网络结构中
# print(np.shape(y_test))
x_train_torch = torch.tensor(x_train,dtype=torch.float)
y_train_torch = torch.tensor(y_train_phi,dtype=torch.float)
x_test_torch = torch.tensor(x_test,dtype=torch.float)
y_test_torch = torch.tensor(y_test_phi,dtype=torch.float)
x_train_torch = x_train_torch.to(device)
y_train_torch = y_train_torch.to(device)
x_test_torch = x_test_torch.to(device)
y_test_torch = y_test_torch.to(device)
torch_dataset = Data.TensorDataset(x_train_torch, y_train_torch)    # 得到一个元组(x, y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size_train,
    shuffle=True,  # 每次训练打乱数据， 默认为False
    num_workers=0,  # 使用多进行程读取数据， 默认0，为不使用多进程
)
for epoch in range(epochs_train):
    for step, (batch_x, batch_y) in enumerate(loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            prediction = model(batch_x) 
            loss = loss_func(prediction, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    if epoch % 100 == 0:
        with torch.no_grad():
                x_train_torch_cpu = x_train_torch.to("cpu")
                y_train_torch_cpu = y_train_torch.to("cpu")
                model_cpu=model
                model_cpu.to("cpu")
                validationprediction=model_cpu(x_train_torch_cpu)
                lossvalidation=loss_func(validationprediction,y_train_torch_cpu)
        if(torch.cuda.is_available()):
            # torch.cuda.empty_cache()
            # with torch.no_grad():
            #     validationprediction=model(x_train_torch)
            #     lossvalidation=loss_func(validationprediction,y_train_torch)
            with torch.no_grad():
                model.to("cuda")
                testprediction=model(x_test_torch)
                losstest=loss_func(testprediction,y_test_torch)
            lossvalidation=lossvalidation.to("cpu")
            losstest=losstest.to("cpu")
            # "train loss",lossvalidation.data.numpy(),
            print(epoch,"train loss",lossvalidation.data.numpy()," test loss",losstest.data.numpy()," R",1-losstest.data.numpy()/ymean_error[i])
    if epoch % 1000 == 0:
        path_save="./networks/model_gpu"+"_"+str(epoch)+"_all_feature.pth"
        torch.save(model.state_dict(),path_save)