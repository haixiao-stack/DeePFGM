import numpy as np
def specialindex(x,distance,choice):
    min=np.min(x)
    max=np.max(x)
    if(choice==1):
        list_spe=np.argwhere(x<min+distance*(max-min))
    else:
        list_spe=np.argwhere(x>min+distance*(max-min))
        print(min+distance*(max-min))
    [a,b]=np.shape(list_spe)
    list_spe=np.reshape(list_spe,a)
    return list_spe
def specialindex2(x,cor1,cor2):
    min=np.min(x)
    max=np.max(x)
    list_spe1=np.argwhere(x>min+cor1*(max-min))
    list_spe2=np.argwhere(x<min+cor2*(max-min))
    list_spe=np.intersect1d(list_spe1,list_spe2) 
    return list_spe
def specialindex3(x,val):
    list_spe=np.argwhere(x>val)
    [a,b]=np.shape(list_spe)
    list_spe=np.reshape(list_spe,a)
    return list_spe
y_T=np.load("./data/raw_data/Ydata_T.npy")
x_T=np.load("./data/raw_data/Xdata_T.npy")
# print(np.shape(x_T))
# print(np.shape(y_T))
list_spe=specialindex3(y_T[:,6],1740)
x_T=x_T[list_spe,:]
y_T=y_T[list_spe,:]
print(np.shape(x_T))
print(np.shape(y_T))
y=np.load("./data/raw_data/Ydata.npy")
x=np.load("./data/raw_data/Xdata.npy")
list_spe=specialindex(y[:,6],0.75,1)
x=x[list_spe,:]
y=y[list_spe,:]
print(np.shape(x))
print(np.shape(y))
# y_large=np.load("./data/raw_data/Ydata_raw_large.npy")
# x_large=np.load("./data/raw_data/Xdata_raw_large.npy")
# list_spe_large=specialindex3(y_large[:,0],8050)
# x_large=x_large[list_spe_large,:]
# y_large=y_large[list_spe_large,:]
Xdata=np.concatenate((x_T,x),axis=0)
# Xdata=np.concatenate((Xdata,x_large),axis=0)
Ydata=np.concatenate((y_T,y),axis=0)
# Ydata=np.concatenate((Ydata,y_large),axis=0)
np.save('./data/raw_data/Xdata_T',Xdata)
np.save('./data/raw_data/Ydata_T',Ydata)