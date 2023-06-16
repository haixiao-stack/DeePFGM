import numpy as np
worktype="data-4D"
name="omegac"
def corindex(x,cor1,cor2,choice):
    min=np.min(x)
    max=np.max(x)
    if(choice==0):
        list_spe=np.argwhere(x<min+cor1*(max-min))
        print(min+cor1*(max-min))
        [a,b]=np.shape(list_spe)
        list_spe=np.reshape(list_spe,a)
    if(choice==1):
        list_spe=np.argwhere(x>min+cor2*(max-min))
        print(min+cor2*(max-min))
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
index=0
value1=2703.4			
value2=5407.5
value3=8111.6
y_add_1=np.load("../../"+worktype+"/raw_data/Ydata_omegac_25.npy")
x_add_1=np.load("../../"+worktype+"/raw_data/Xdata_omegac_25.npy")
list_spe1 = valueindex(y_add_1[:,index],value1,value2,2)
list_spe1 = np.random.choice(list_spe1, size=1000000, replace=False)
y_add_1=y_add_1[list_spe1,:]
x_add_1=x_add_1[list_spe1,:]
y_add_2=np.load("../../"+worktype+"/raw_data/Ydata_omegac_50.npy")
x_add_2=np.load("../../"+worktype+"/raw_data/Xdata_omegac_50.npy")
list_spe2 = valueindex(y_add_2[:,index],value2,value3,2)
list_spe2 = np.random.choice(list_spe2, size=1000000, replace=True)
y_add_2=y_add_2[list_spe2,:]
x_add_2=x_add_2[list_spe2,:]
y_add_3=np.load("../../"+worktype+"/raw_data/Ydata_omegac_75.npy")
x_add_3=np.load("../../"+worktype+"/raw_data/Xdata_omegac_75.npy")
list_spe3 = valueindex(y_add_3[:,index],value2,value3,1)
list_spe3 = np.random.choice(list_spe3, size=1200000, replace=True)
y_add_3=y_add_3[list_spe3,:]
x_add_3=x_add_3[list_spe3,:]
# list_spe=np.concatenate((list_spe1,list_spe2),axis=0)
# list_spe3 = valueindex(y_add[:,index],value2,value3,1)
# y_add=y_add[list_spe,:]
# x_add=x_add[list_spe,:]
# list_spe3 = np.random.choice(list_spe3, size=150000, replace=True)
# list_spe=np.concatenate((list_spe,list_spe3),axis=0) 
# list_spe3 = valueindex(y_add[:,0],5490,8234,1)
# list_spe = np.concatenate((list_spe,list_spe3),axis=0) 
# print(np.shape(list_spe1))
# print(np.shape(list_spe2))
# print(np.shape(list_spe3))
# print(np.shape(x_T))
# print(np.shape(y_T))
# list_spe1 = valueindex(y_add[:,8],0.095,0.115,2)
# list_spe1 = np.random.choice(list_spe1, size=150000, replace=False)
# list_spe2 = valueindex(y_add[:,8],0.095,0.115,1)
# list_spe2 = np.random.choice(list_spe2, size=100000, replace=False)
# list_spe=np.concatenate((list_spe1,list_spe2),axis=0) 
# y_add=y_add[list_spe,:]
# x_add=x_add[list_spe,:]
# print(np.shape(x_add))
print(np.shape(y_add_1))
y=np.load("../../"+worktype+"/raw_data/Ydata_all.npy")
x=np.load("../../"+worktype+"/raw_data/Xdata_all.npy")
# list_spe1 = corindex(y[:,10],0.25,0.75,1)
# print(np.shape(list_spe1))
# # list_spe1 = corindex(y[:,9],0.25,0.75,1)
# # print(np.shape(list_spe1))
# list_spe=corindex(y[:,9],0.75,1,0)
# x=x[list_spe,:]
# y=y[list_spe,:]
# print(np.shape(x))
print(np.shape(y))
# y_large=np.load("./data/raw_data/Ydata_raw_large.npy")
# x_large=np.load("./data/raw_data/Xdata_raw_large.npy")
# list_spe_large=specialindex3(y_large[:,0],8050)
# x_large=x_large[list_spe_large,:]
# y_large=y_large[list_spe_large,:]
Xdata=np.concatenate((x_add_1,x),axis=0)
Xdata=np.concatenate((x_add_2,Xdata),axis=0)
Xdata=np.concatenate((x_add_3,Xdata),axis=0)
# Xdata=np.concatenate((Xdata,x_large),axis=0)
Ydata=np.concatenate((y_add_1,y),axis=0)
Ydata=np.concatenate((y_add_2,Ydata),axis=0)
Ydata=np.concatenate((y_add_3,Ydata),axis=0)
# list_spe3 = valueindex(Ydata[:,8],0.095,0.115,1)
# print(np.shape(list_spe3))
# print(np.shape(Xdata))
list_spe1 = valueindex(Ydata[:,index],value2,value3,1)
print(np.shape(list_spe1))
print(np.shape(Ydata))
# Ydata=np.concatenate((Ydata,y_large),axis=0)
np.save('../../'+worktype+'/raw_data/Xdata_'+name,Xdata)
np.save('../../'+worktype+'/raw_data/Ydata_'+name,Ydata)