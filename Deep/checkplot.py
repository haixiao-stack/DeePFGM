#!/usr/bin/env python
# coding: utf-8

# In[13]:

# matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
import json
def locate(xarray, n, x):
    if (x < xarray[1]):
        return 1
    if (x >= xarray[n]):
        return n - 1
    for k in range(1, n):
        if (x >= xarray[k] and x < xarray[k + 1]): return k
# In[14]:
f = open('./data/plot.json', 'r')
content = f.read()
a = json.loads(content)
f.close()
cbDict_draw=a['plot']
file=open('./data/flare.tbl')    
xaxis_name=cbDict_draw['Xaxis'].lower()
yaxis_name=cbDict_draw['Yaxis'].lower()
dataMat=[]  
labelMat=[]
floatLines=[]
for line in file.readlines():    
    curLine=line.strip().split("\t")
    curLine=np.array(curLine)
    floatLine=curLine.astype(np.float64)#这里使用的是map函数直接把数据转化成为float类型    
    floatLines.append(floatLine)
#     dataMat.append(floatLine[0:2]) 
#     labelMat.append(floatLine[-1])
# print("dataMat:",dataMat)
# print("labelMat:",labelMat)
rownum=1
chemnum=np.int32(floatLines[rownum][0])
for i in range(chemnum):
    rownum=rownum+1
    if(i==0):
        chemvalues=floatLines[rownum]
    else:
        chemvalues=np.append(chemvalues,floatLines[rownum])
x=np.shape(floatLines[rownum])
x=np.int32(x)
chemvalues=np.reshape(chemvalues,(chemnum,x[0]))
rownum=rownum+1
znum=np.int32(floatLines[rownum][0])
for i in range(znum):
    rownum=rownum+1
    if(i==0):
        zvalues=floatLines[rownum]
    else:
        zvalues=np.append(zvalues,floatLines[rownum])
rownum=rownum+1
cnum=np.int32(floatLines[rownum][0])
for i in range(cnum):
    rownum=rownum+1
    if(i==0):
        cvalues=floatLines[rownum]
    else:
        cvalues=np.append(cvalues,floatLines[rownum])
rownum=rownum+1
gznum=np.int32(floatLines[rownum][0])
for i in range(gznum):
    rownum=rownum+1
    if(i==0):
        gzvalues=floatLines[rownum]
    else:
        gzvalues=np.append(gzvalues,floatLines[rownum])
rownum=rownum+1
gcnum=np.int32(floatLines[rownum][0])
for i in range(gcnum):
    rownum=rownum+1
    if(i==0):
        gcvalues=floatLines[rownum]
    else:
        gcvalues=np.append(gcvalues,floatLines[rownum])
rownum=rownum+1
gzcnum=np.int32(floatLines[rownum][0])
for i in range(gzcnum):
    rownum=rownum+1
    if(i==0):
        gzcvalues=floatLines[rownum]
    else:
        gzcvalues=np.append(gzcvalues,floatLines[rownum])
rownum=rownum+1
nScalars=np.int32(floatLines[rownum][0])
nYis=np.int32(floatLines[rownum][1])
rownum=rownum+1
# datavalues = floatLines[rownum:rownum+gzcnum*gcnum*gznum*gcnum*gznum]
datavalues = floatLines[rownum:rownum+gzcnum*gcnum*gznum*cnum*znum]


# In[15]:


phinum=cbDict_draw['phinum']
datavalues=np.array(datavalues)
phi=datavalues[:,phinum]
prediction=np.load("./data/predictions/T.npy")
# print(phi[245390])
# datavalues=np.reshape(datavalues,(gzcnum*gcnum*gznum*cnum*znum,nScalars+nYis))
# phi=datavalues[:,phinum]
Q = np.reshape(phi,(znum,cnum,gznum,gcnum,gzcnum))
prediction = np.reshape(prediction,(znum,cnum,gznum,gcnum,gzcnum))
error=(prediction-Q)/Q
nums = {'z': znum, 'c': cnum, 'gz': gznum,'gc': gcnum}
values = {'z': zvalues, 'c': cvalues, 'gz': gzvalues,'gc': gcvalues}
indexes={'z': 0, 'c': 0, 'gz': 0,'gc': 0}
fix_name1=cbDict_draw['fixname1']
fix_name2=cbDict_draw['fixname2']
fix_value1=cbDict_draw['fixvalue1']
fix_value2=cbDict_draw['fixvalue2']
indexes[cbDict_draw['fixname1']]=locate(values[cbDict_draw['fixname1']], len(values[cbDict_draw['fixname1']])-1, cbDict_draw['fixvalue1'])
indexes[cbDict_draw['fixname2']]=locate(values[cbDict_draw['fixname2']], len(values[cbDict_draw['fixname2']])-1, cbDict_draw['fixvalue2'])
# In[16]:
xvalues=values[xaxis_name]
yvalues=values[yaxis_name]
xnum=nums[xaxis_name]
ynum=nums[yaxis_name]
plotvalues=np.zeros((ynum,xnum))
X, Y = np.meshgrid(xvalues, yvalues)
# print(np.shape(X))
# print(np.shape(plotvalues))
for i in range(ynum):
    for j in range(xnum):
        indexes[xaxis_name]=j
        indexes[yaxis_name]=i
        plotvalues[i][j]=error[indexes['z']][indexes['c']][indexes['gz']][indexes['gc']][0]
        # print(i," ",j)
#填充等高线
X.astype('float32').tofile('X.dat', sep=' ')
Y.astype('float32').tofile('Y.dat', sep=' ')
plotvalues.astype('float32').tofile('values.dat', sep=' ')
nums=[ynum,xnum]
nums=np.array(nums)
nums.astype('int').tofile('nums.dat', sep=' ')
names=[xaxis_name,yaxis_name]
filename = open("names.dat", "w")
for a in names:
    filename.write((a+" "))
filename.close()
data = np.fromfile('X.dat', dtype='float32', sep=' ')
X = data.reshape((ynum, xnum))
data = np.fromfile('Y.dat', dtype='float32', sep=' ')
Y = data.reshape((ynum, xnum))
data = np.fromfile('values.dat', dtype='float32', sep=' ')
plotvalues = data.reshape((ynum, xnum))
plt.contourf(X, Y, plotvalues, levels=10, cmap='RdBu')
plt.colorbar()
plt.xlabel(xaxis_name)
plt.ylabel(yaxis_name)
plt.show()
plt.savefig('contourf.png')
# plt.xlim(0,0.1)
# 显示图表
# print(values)


# In[ ]:





# In[12]:





# In[ ]:




