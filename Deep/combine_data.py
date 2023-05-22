import numpy as np
def corindex(x,cor1,cor2,choice):
    min=np.min(x)
    max=np.max(x)
    if(choice==0):
        list_spe=np.argwhere(x<min+cor1*(max-min))
        [a,b]=np.shape(list_spe)
        list_spe=np.reshape(list_spe,a)
    if(choice==1):
        list_spe=np.argwhere(x>min+cor2*(max-min))
        [a,b]=np.shape(list_spe)
        list_spe=np.reshape(list_spe,a)
    if(choice==2):
        list_spe1=np.argwhere(x>min+cor1*(max-min))
        list_spe2=np.argwhere(x<min+cor2*(max-min))
        list_spe=np.intersect1d(list_spe1,list_spe2) 
    return list_spe
def valueindex(x,val1,val2,choice):
    if(choice==0):
        list_spe=np.argwhere(x<val1)
        [a,b]=np.shape(list_spe)
        list_spe=np.reshape(list_spe,a)
    if(choice==1):
        list_spe=np.argwhere(x>val2)
        [a,b]=np.shape(list_spe)
        list_spe=np.reshape(list_spe,a)
    if(choice==2):
        list_spe1=np.argwhere(x>val1)
        list_spe2=np.argwhere(x<val2)
        list_spe=np.intersect1d(list_spe1,list_spe2) 
    return list_spe
y_add=np.load("../../data/raw_data/Ydata_YH2O.npy")
x_add=np.load("../../data/raw_data/Xdata_YH2O.npy")
# print(np.shape(x_T))
# print(np.shape(y_T))
list_spe1 = valueindex(y_add[:,8],0.095,0.115,2)
list_spe1 = np.random.choice(list_spe1, size=150000, replace=False)
list_spe2 = valueindex(y_add[:,8],0.095,0.115,1)
list_spe2 = np.random.choice(list_spe2, size=100000, replace=False)
list_spe=np.concatenate((list_spe1,list_spe2),axis=0) 
x_add=x_add[list_spe,:]
y_add=y_add[list_spe,:]
print(np.shape(x_add))
print(np.shape(y_add))
y=np.load("../../data/raw_data/Ydata.npy")
x=np.load("../../data/raw_data/Xdata.npy")
list_spe=corindex(y[:,8],0.75,1,0)
x=x[list_spe,:]
y=y[list_spe,:]
print(np.shape(x))
print(np.shape(y))
# y_large=np.load("./data/raw_data/Ydata_raw_large.npy")
# x_large=np.load("./data/raw_data/Xdata_raw_large.npy")
# list_spe_large=specialindex3(y_large[:,0],8050)
# x_large=x_large[list_spe_large,:]
# y_large=y_large[list_spe_large,:]
Xdata=np.concatenate((x_add,x),axis=0)
# Xdata=np.concatenate((Xdata,x_large),axis=0)
Ydata=np.concatenate((y_add,y),axis=0)
list_spe3 = valueindex(Ydata[:,8],0.095,0.115,1)
print(np.shape(list_spe3))
print(np.shape(Xdata))
# Ydata=np.concatenate((Ydata,y_large),axis=0)
np.save('../../data/raw_data/Xdata_YH2O',Xdata)
np.save('../../data/raw_data/Ydata_YH2O',Ydata)