import numpy as np
import time
import multiprocessing as mp
from scipy.stats import beta
import scipy.io as sio
from math import exp
import os
def readcopula(filename):
    roadef_info = sio.loadmat(filename)
    prob = roadef_info['y'][0]
    return prob
def processdata(data,k,x,y):
    data_new=[]
    for i in range(0,k):
        data_new.append(data[:,:,i])
    data_new=np.array(data_new)
    return data_new
def computedata(data,type,n):
    res = np.zeros_like(data)
    if(type=="average"):
        res[1:n]=(data[1:n]+data[2:n+1])/2
    if(type=="diff"):
        res[1:n]=data[2:n+1]-data[1:n]
    return res
def locate(xarray,n,x):
    if(x<xarray[1]):
        return 1
    if(x>=xarray[n]):
        return n-1
    for k in range(1,n):
        if(x>=xarray[k] and x<xarray[k+1]): return k
def intfac(x,xarray,loc_low):
    if(x<xarray[loc_low]): return 0
    if (x > xarray[loc_low+1]): return 1
    return (x-xarray[loc_low])/(xarray[loc_low+1]-xarray[loc_low])
def Psicomputing(sc_vals,sc_vals1,sc_vals2,sc_valsScalars,c_space,z_space,n_points_z,n_points_c,nScalars,nYis,k):
    z_space_use = np.reshape(np.repeat(z_space, n_points_c + 1), (n_points_z + 1, n_points_c + 1))
    c_space_use = np.reshape(np.tile(c_space, n_points_z + 1), (n_points_z + 1, n_points_c + 1))
    if(k<2 or k>4): return sc_vals[1:n_points_z+1,1:n_points_c + 1]
    if(k==2): return sc_vals[1:n_points_z+1,1:n_points_c + 1]/ sc_vals1[1:n_points_z+1,1:n_points_c + 1]
    if(k==3): return np.multiply(np.multiply(np.multiply(sc_vals2[1:n_points_z+1,1:n_points_c + 1], 1/ sc_vals1[1:n_points_z+1,1:n_points_c + 1]), c_space_use[1:n_points_z+1,1:n_points_c + 1]), sc_valsScalars[1:n_points_z+1,1:n_points_c + 1])
    if(k==4): return np.multiply(np.multiply(sc_vals2[1:n_points_z+1,1:n_points_c + 1], 1 / sc_vals1[1:n_points_z+1,1:n_points_c + 1]), z_space_use[1:n_points_z+1,1:n_points_c + 1])
def c_dYdccomputing(sc_vals_int,sc_vals_int1,sc_vals_int2,sc_vals_intScalars,space,space_diff,space_average,mean,cdf001,n_points_c,index):
    dsdc=np.zeros(n_points_c+2)
    Yc_0 = np.zeros(n_points_c + 2)
    Yc_1 = np.zeros(n_points_c + 2)
    if(index<2 or index>4): dsdc[1:n_points_c] = (sc_vals_int[2:n_points_c + 1] - sc_vals_int[1:n_points_c]) / space_diff[1:n_points_c]
    if(index==2): dsdc[1:n_points_c] = (sc_vals_int2[2:n_points_c + 1] / sc_vals_int1[2:n_points_c + 1] - sc_vals_int2[1:n_points_c] / sc_vals_int1[1:n_points_c]) / space_diff[1:n_points_c]
    if(index==3):
        Yc_0[1:n_points_c] = space[1:n_points_c] * sc_vals_intScalars[1:n_points_c]
        Yc_1[2:n_points_c+1] = space[2:n_points_c + 1] * sc_vals_intScalars[2:n_points_c + 1]
        dsdc[1:n_points_c] = (np.multiply(Yc_1[2:n_points_c+1],sc_vals_int2[2:n_points_c + 1] / sc_vals_int1[2:n_points_c + 1]) - np.multiply(Yc_0[1:n_points_c], sc_vals_int2[1:n_points_c] / sc_vals_int1[1:n_points_c])) / space_diff[1:n_points_c]
    if(index==4):
        dsdc[1:n_points_c] = (mean * sc_vals_int2[2:n_points_c + 1] / sc_vals_int1[2:n_points_c + 1] - mean*sc_vals_int2[1:n_points_c] /sc_vals_int1[1:n_points_c]) / space_diff[1:n_points_c]
    dsdc[n_points_c] = dsdc[n_points_c - 1]
    dsdc[n_points_c + 1] = dsdc[1]
    y_int=0
    y_int = y_int - np.sum(0.5 * np.multiply((np.multiply(dsdc[1:n_points_c - 1], cdf001[1:n_points_c - 1]) + np.multiply(dsdc[2:n_points_c],cdf001[2:n_points_c])),(space_average[2:n_points_c] - space_average[1:n_points_c - 1])))
    y_int = y_int - dsdc[n_points_c] * cdf001[n_points_c] * (space[n_points_c] - space[n_points_c - 1]) / 2.0 - dsdc[n_points_c + 1] * cdf001[1] * (space[2] - space[1]) / 2.0 + sc_vals_int[n_points_c]
    return y_int
def z_dYdccomputing(sc_vals_int,sc_vals_int1,sc_vals_int2,sc_vals_intScalars,space,space_diff,space_average,mean,cdf001,n_points_z,index):
    dsdc=np.zeros(n_points_z+2)
    Yc_0 = np.zeros(n_points_z + 2)
    Yc_1 = np.zeros(n_points_z + 2)
    if(index<2 or index>4): dsdc[1:n_points_z] = (sc_vals_int[2:n_points_z + 1] - sc_vals_int[1:n_points_z]) / space_diff[1:n_points_z]
    if(index==2): dsdc[1:n_points_z] = (sc_vals_int2[2:n_points_z + 1] / sc_vals_int1[2:n_points_z + 1] - sc_vals_int2[1:n_points_z] / sc_vals_int1[1:n_points_z]) / space_diff[1:n_points_z]
    if(index==3):
        Yc_0[1:n_points_z] = mean * sc_vals_intScalars[1:n_points_z]
        Yc_1[2:n_points_z+1] = mean * sc_vals_intScalars[2:n_points_z + 1]
        dsdc[1:n_points_z] = (np.multiply(Yc_1[2:n_points_z+1],sc_vals_int2[2:n_points_z + 1] / sc_vals_int1[2:n_points_z + 1]) - np.multiply(Yc_0[1:n_points_z], sc_vals_int2[1:n_points_z] / sc_vals_int1[1:n_points_z])) / space_diff[1:n_points_z]
    if(index==4):
        dsdc[1:n_points_z] = (space[2:n_points_z + 1] * sc_vals_int2[2:n_points_z + 1] / sc_vals_int1[2:n_points_z + 1] - space[1:n_points_z]*sc_vals_int2[1:n_points_z] /sc_vals_int1[1:n_points_z]) / space_diff[1:n_points_z]
    dsdc[n_points_z] = dsdc[n_points_z - 1]
    dsdc[n_points_z + 1] = dsdc[1]
    y_int=0
    y_int = y_int - np.sum(0.5 * np.multiply((np.multiply(dsdc[1:n_points_z - 1], cdf001[1:n_points_z - 1]) + np.multiply(dsdc[2:n_points_z],cdf001[2:n_points_z])),(space_average[2:n_points_z] - space_average[1:n_points_z - 1])))
    y_int = y_int - dsdc[n_points_z] * cdf001[n_points_z] * (space[n_points_z] - space[n_points_z - 1]) / 2.0 - dsdc[n_points_z + 1] * cdf001[1] * (space[2] - space[1]) / 2.0 + sc_vals_int[n_points_z]
    return y_int
def pdfFunc(z_space, c_space,alpha_z, beta_z, alpha_c, beta_c, n_points_c,n_points_z,type,rho,parameters):
    CDF_C = np.zeros(n_points_c + 2)
    CDF_Z = np.zeros(n_points_z + 2)
    CDF_Z[1:n_points_z + 1] = beta.cdf(z_space[1:n_points_z + 1], alpha_z, beta_z)
    CDF_C[1:n_points_c + 1] = beta.cdf(c_space[1:n_points_c + 1], alpha_c, beta_c)
    print(CDF_Z)
    # j = n_points_c
    # CDF_C[j] = beta.cdf((c_space[j - 1] + 3 * c_space[j]) / 4.0, alpha_c, beta_c)
    # i = n_points_z
    # CDF_Z[i] = beta.cdf((z_space[i - 1] + 3 * z_space[i]) / 4.0, alpha_z, beta_z)
    if (rho == 0): type = "independent"
    X,Y = np.meshgrid(CDF_C, CDF_Z)
    if (type == "independent"): 
        CDF_multi=np.multiply(X,Y)
        return CDF_C, CDF_Z, CDF_multi
    if (type == "frank"):
        frankparameters=parameters['frank']
        index = int((rho + 1) / 0.01)
        if (index == 200):
            alpha = frankparameters[index]
        elif (round(rho, 2) >= rho):
            alpha = ((round(rho, 2) - rho) * frankparameters[index + 1] + (rho - round(rho, 2) + 0.01) *
                     frankparameters[index]) * 100
        else:
            alpha = ((rho - round(rho, 2)) * frankparameters[index + 1] + (round(rho, 2) - rho + 0.01) *
                     frankparameters[index]) * 100
        if (alpha > 35):   alpha = 35
        CDF_multi = -(1/alpha)*np.log(1+(np.multiply((np.exp(-alpha*X)-1),(np.exp(-alpha*Y)-1)))/(np.exp(-alpha)-1))
        return CDF_C, CDF_Z,CDF_multi
def c_cdfFunc(space_average, alpha_c, beta_c, n_points_c):
    cdf001 = beta.cdf(space_average, alpha_c, beta_c)
    cdf001[n_points_c] = 1
    return cdf001
def z_cdfFunc(space_average, alpha_z, beta_z, n_points_z):
    cdf001 = beta.cdf(space_average, alpha_z, beta_z)
    cdf001[n_points_z] = 1
    return cdf001
def delta(z_mean,c_mean,z_space,c_space,sc_vals,Yi_vals,n_points_z,n_points_c,nScalars,nYis):
    y_int=np.zeros(nScalars+1)
    Yi_int=np.zeros(nYis+1)
    z_loc = locate(z_space, n_points_z, z_mean)
    z_fac = intfac(z_mean, z_space, z_loc)
    c_loc = locate(c_space, n_points_c, c_mean)
    c_fac = intfac(c_mean, c_space, c_loc)
    for i in range(1,3):
        y_int[i]=(1-c_fac) * (z_fac * sc_vals[i][z_loc + 1][c_loc] +(1-z_fac) * (sc_vals[i][z_loc][c_loc]))+ c_fac * (z_fac * sc_vals[i][z_loc + 1][c_loc+1] + (1-z_fac) * sc_vals[i][z_loc][c_loc+1])
    for i in range(5,nScalars+1):
        y_int[i] = (1 - c_fac) * (z_fac * sc_vals[i][z_loc + 1][c_loc] + (1 - z_fac) * (sc_vals[i][z_loc][c_loc])) + c_fac * (z_fac * sc_vals[i][z_loc + 1][c_loc+1] + (1 - z_fac) * sc_vals[i][z_loc][c_loc+1])
    y_int[2] = y_int[2]/ y_int[1] #Zhi: omega_c / rho
    y_int[3] = y_int[2] * c_mean * y_int[nScalars] # gc_source
    y_int[4] = y_int[2] * z_mean # gz_source
    for i in range(1,nYis+1):
        Yi_int[i] = (1 - c_fac) * (z_fac * Yi_vals[i][z_loc + 1][c_loc] + (1 - z_fac) * (Yi_vals[i][z_loc][c_loc])) + c_fac * (z_fac * Yi_vals[i][z_loc + 1][c_loc + 1] + (1 - z_fac) * Yi_vals[i][z_loc][c_loc + 1])
    y_int = np.where(abs(y_int) > 1e-20, y_int, 0)
    Yi_int = np.where(Yi_int > 1e-20, Yi_int, 0)
    return y_int,Yi_int
def cbeta(mean,g_var,space,space_average,space_diff,z_mean,z_space,sc_vals,Yi_vals,n_points_z,n_points_c,nScalars,nYis):
    loc = locate(z_space, n_points_z, z_mean)
    fac = intfac(z_mean, z_space, loc)
    sc_vals_int=np.zeros((nScalars+1,n_points_c+1))
    Yi_vals_int=np.zeros((nYis+1,n_points_c+1))
    y_int = np.zeros(nScalars + 1)
    Yi_int = np.zeros(nYis + 1)
    alpha_c = mean * ((1 / g_var) - 1)
    beta_c = (1 - mean) * ((1 / g_var) - 1)
    sc_vals_int[1:nScalars+1,1:n_points_c+1]=fac * sc_vals[1:nScalars+1,loc+1,1:n_points_c+1]+(1 - fac) * sc_vals[1:nScalars+1,loc,1:n_points_c+1]
    Yi_vals_int[1:nYis + 1, 1:n_points_c + 1] = fac * Yi_vals[1:nYis + 1, loc + 1, 1:n_points_c + 1] + (1 - fac) * Yi_vals[1:nYis + 1, loc, 1:n_points_c + 1]
    cdf001=c_cdfFunc(space_average,alpha_c,beta_c,n_points_c)
    sc_vals_int1 = sc_vals_int[1]
    sc_vals_int2 = sc_vals_int[2]
    sc_vals_intScalars = sc_vals_int[nScalars]
    for j in range(1,nScalars+1):
        y_int[j]=c_dYdccomputing(sc_vals_int[j], sc_vals_int1, sc_vals_int2, sc_vals_intScalars, space, space_diff, space_average, z_mean, cdf001, n_points_c, j)
    for j in range(1,nYis+1):
        Yi_int[j]=c_dYdccomputing(Yi_vals_int[j], sc_vals_int1, sc_vals_int2, sc_vals_intScalars, space, space_diff, space_average, z_mean, cdf001, n_points_c, 1)
    y_int = np.where(abs(y_int) > 1e-20, y_int, 0)
    Yi_int = np.where(Yi_int > 1e-20, Yi_int, 0)
    return y_int,Yi_int
def zbeta(mean,g_var,space,space_average,space_diff,c_mean,c_space,sc_vals,Yi_vals,n_points_z,n_points_c,nScalars,nYis):
    loc = locate(c_space, n_points_c, c_mean)
    fac = intfac(c_mean, c_space, loc)
    sc_vals_int=np.zeros((nScalars+1,n_points_z+1))
    Yi_vals_int=np.zeros((nYis+1,n_points_z+1))
    y_int = np.zeros(nScalars + 1)
    Yi_int = np.zeros(nYis + 1)
    alpha_z = mean * ((1 / g_var) - 1)
    beta_z = (1 - mean) * ((1 / g_var) - 1)
    sc_vals_int[1:nScalars + 1, 1:n_points_z + 1] = fac * sc_vals[1:nScalars + 1, 1:n_points_z + 1, loc + 1] + (1 - fac) * sc_vals[1:nScalars + 1, 1:n_points_z + 1,loc]
    Yi_vals_int[1:nYis + 1, 1:n_points_z + 1] = fac * Yi_vals[1:nYis + 1, 1:n_points_z + 1,loc + 1] + (1 - fac) * Yi_vals[1:nYis + 1,1:n_points_z + 1,loc]
    cdf001 = z_cdfFunc(space_average, alpha_z, beta_z, n_points_z)
    sc_vals_int1 = sc_vals_int[1]
    sc_vals_int2 = sc_vals_int[2]
    sc_vals_intScalars = sc_vals_int[nScalars]
    for j in range(1,nScalars+1):
        y_int[j] = z_dYdccomputing(sc_vals_int[j], sc_vals_int1, sc_vals_int2, sc_vals_intScalars, space, space_diff,space_average, c_mean, cdf001, n_points_z, j)
    for j in range(1,nYis+1):
        Yi_int[j] = z_dYdccomputing(Yi_vals_int[j], sc_vals_int1, sc_vals_int2, sc_vals_intScalars, space, space_diff,space_average, c_mean, cdf001, n_points_z, 1)
    y_int=np.where(abs(y_int) > 1e-20, y_int, 0)
    Yi_int=np.where(Yi_int > 1e-20, Yi_int, 0)
    return y_int, Yi_int
def int_point(z_mean,c_mean,c_var,z_var,rho,z_space,c_space,z_space_average,c_space_average,Psi_compute,YiPsi_compute,n_points_z,n_points_c,nScalars,nYis,type,parameters):
    y_int=np.zeros(nScalars+1)
    Yi_int=np.zeros(nYis+1)
    alpha_z = z_mean * (((z_mean * (1-z_mean)) / z_var) - 1)
    alpha_c = c_mean * (((c_mean * (1-c_mean)) / c_var) - 1)
    beta_z = (1-z_mean) * (((z_mean * (1-z_mean)) / z_var) - 1)
    beta_c = (1-c_mean) * (((c_mean * (1-c_mean)) / c_var) - 1)
    print(alpha_z,alpha_c,beta_z,beta_c)
    CDF_C, CDF_Z, CDF_multi = pdfFunc(z_space, c_space,alpha_z, beta_z, alpha_c, beta_c, n_points_c, n_points_z,type,rho,parameters)
    CDF_multi_compute=np.zeros((n_points_z-1,n_points_c-1))
    CDF_multi_compute=(CDF_multi[1:n_points_z,1:n_points_c]+CDF_multi[2:n_points_z+1,2:n_points_c+1]-CDF_multi[1:n_points_z,2:n_points_c+1]-CDF_multi[2:n_points_z+1,1:n_points_c])
    for k in range(2,nScalars+1):
        y_int[k] = np.sum(np.multiply(Psi_compute[k],CDF_multi_compute))
    for k in range(1,nYis+1):
        Yi_int[k] = np.sum(np.multiply(YiPsi_compute[k],CDF_multi_compute))
    y_int = np.where(abs(y_int) > 1e-20, y_int, 0)
    Yi_int = np.where(Yi_int > 1e-20, Yi_int, 0)
    return y_int,Yi_int
def readdata(cbDict,filename):
    length=int(cbDict['int_pts_c']*cbDict['int_pts_gc']*cbDict['int_pts_gz'])
    Y_ints=np.zeros((length+1,cbDict['nScalars']))
    Yi_ints=np.zeros((length+1,cbDict['nYis'] + 1))
    start=time.time()
    f = open(filename)
    for i in range(1,length+1):
        data=f.readline()
        data=data.split()
        Y_ints[i][1:]=[eval(x) for x in data[5:cbDict['nScalars']+4]]
        Yi_ints[i][1:]=[eval(x) for x in data[cbDict['nScalars']+4:]]
    print("Reading data耗时", time.time() - start, "s")
    print("文件名称",filename)
    return Y_ints,Yi_ints
def readspace(cbDict,ih):
    z_space = np.zeros(cbDict['n_points_z'] + 1)
    c_space = np.zeros(cbDict['n_points_c'] + 1)
    Src_vals = np.zeros((cbDict['n_points_z'] + 1, cbDict['n_points_c'] + 1, cbDict['nScalars'] + 1))
    Yi_vals = np.zeros((cbDict['n_points_z'] + 1, cbDict['n_points_c'] + 1, cbDict['nYis'] + 1))
    str_ih='%02d'%ih
    start1 = time.time()
    print('Reading chemTab...')
    f = open('./canteraData/chemTab_'+str_ih+'.dat')
    for i in range(1,cbDict['n_points_z']+1):
        for j in range(1,cbDict['n_points_c']+1):
            data=f.readline()
            data=data.split()
            z_space[i]=eval(data[0])
            c_space[j]=eval(data[1])
            Src_vals[i][j][1:]=[eval(x) for x in data[2:cbDict['nScalars']+2]]
            Yi_vals[i][j][1:] = [eval(x) for x in data[cbDict['nScalars'] + 2:]]
    print('Reading done')
    print("Reading chemTab耗时", time.time() - start1, "s")
    start=time.time()
    Src_vals=processdata(Src_vals,cbDict['nScalars']+1,cbDict['n_points_z']+1,cbDict['n_points_c']+1)
    Yi_vals = processdata(Yi_vals, cbDict['nYis'] + 1, cbDict['n_points_z'] + 1, cbDict['n_points_c'] + 1)
    return z_space,c_space,Src_vals,Yi_vals
def integrate(cbDict,z_space,z_int,c_space,c_int,Src_vals,Yi_vals,gz_int,gc_int,gcz_int,iz,igcz,ih,parameters,Psi_compute,YiPsi_compute):
      start = time.time()
      yint = np.zeros(cbDict['nScalars'] + 1)
      Yi_int = np.zeros(cbDict['nYis'] + 1)
      p=0
      str_iz='%02d' % iz
      str_ih='%02d' % ih
      rho=gcz_int[igcz]
      str_rho = '%.1f' % rho
      z_space_average = computedata(z_space, "average", cbDict['n_points_z'])
      z_space_diff = computedata(z_space, "diff", cbDict['n_points_z'])
      c_space_average = computedata(c_space, "average", cbDict['n_points_c'])
      c_space_diff = computedata(c_space, "diff", cbDict['n_points_c'])
    #   print(cbDict["pdf_type"])
      if(cbDict["pdf_type"]=="independent"):
        f = open('unit'+str_iz+'_h'+str_ih+'.dat', "w")
      else:
        f = open('unit'+str_iz+'_h'+str_ih+'_rho_'+str_rho+'.dat', "w")
      z_loc = locate(z_space, cbDict['n_points_z'], z_int[iz])
      for ic in range(5,6):#1,cbDict['int_pts_c']+1
        for igz in range(7,8):#1,cbDict['int_pts_gz']+1
          for igc in range(9,10):#1,cbDict['int_pts_gc']+1
              p=p+1
              print("computing unit"+str_iz+" case ",p," ",iz," ",ic," ",igz," ",igc," ",igcz)#显示计算进程，嫌烦这行可以删除
              if((iz==1) or (iz==cbDict['int_pts_z'])):
                yint[2]=0
                yint[3]=0
                yint[4]=0
                if (iz == 1): z_loc=1
                if (iz == cbDict['int_pts_z']): z_loc=cbDict['n_points_z']
                for i in range(5,cbDict['nScalars']+1):
                  yint[i]=Src_vals[i][z_loc][1]
                for i in range(1,cbDict['nYis']+1):
                  Yi_int[i]=Yi_vals[i][z_loc][1]
              elif(((igz == 1 and igc == 1) or (igz == 1 and ic == 1)) or (igz == 1 and ic == cbDict['int_pts_c'])):
                yint,Yi_int=delta(z_int[iz], c_int[ic], z_space, c_space,Src_vals,Yi_vals,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'])
              elif(igz==1 and igc>1):
                yint,Yi_int =cbeta(c_int[ic],gc_int[igc],c_space,c_space_average,c_space_diff,z_int[iz],z_space,Src_vals,Yi_vals,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'])
              elif(igz>1 and igc==1):
                yint,Yi_int =zbeta(z_int[iz],gz_int[igz],z_space,z_space_average,z_space_diff,c_int[ic],c_space,Src_vals,Yi_vals,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'])
              else:
                  k=0
                  if((ic == 1) or (ic == cbDict['int_pts_c'])): k=1
                  if(k==0):
                    c_var=gc_int[igc]*(c_int[ic]*(1.0-c_int[ic]))
                    z_var=gz_int[igz]*(z_int[iz]*(1.0-z_int[iz]))
                    yint,Yi_int=int_point(z_int[iz],c_int[ic],c_var,z_var,gcz_int[igcz],z_space,c_space,z_space_average,c_space_average,Psi_compute,YiPsi_compute,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'],cbDict['pdf_type'],parameters)#cbDict['pdf_type']
              data=[z_int[iz],c_int[ic],gz_int[igz],gc_int[igc],gcz_int[igcz]]+[x for x in yint[2:(cbDict['nScalars']+1)]]+[x for x in Yi_int[1:]]
              for i in range(0,len(data)):
                f.write(str('%0.5E' % data[i]))
                f.write(" ")
              f.write("\n")
      print("Writing done ",'unit'+str_iz+'_h'+str_ih+'_rho_'+str_rho+'.dat')
      print("写该文件总耗时"," ",time.time()-start,"s")
      f.close()
def pdf_multi(item):
    cbDict,iz,igcz,ih,z_space,c_space,Src_vals,Yi_vals,Psi_compute,YiPsi_compute=item
    parameters = {}
    parameters['frank'] = readcopula("./frankparameters.mat")
    parameters['placket'] = readcopula("./placketparameters.mat")
    integrate(cbDict,z_space,cbDict['z'],c_space,cbDict['c'],Src_vals,Yi_vals,cbDict['gz'],cbDict['gc'],cbDict['gcz'],iz,igcz,ih,parameters,Psi_compute,YiPsi_compute)
def multiprocessingpdf(cbDict):
    start=time.time()
    pool=mp.Pool(processes=cbDict['n_procs'])
    pool2=mp.Pool(processes=cbDict['n_procs'])
    paramlists=[]
    ih=cbDict['n_points_h']
    z_space,c_space,Src_vals,Yi_vals=readspace(cbDict,ih)
    Psi=np.zeros((cbDict['nScalars']+1,cbDict['n_points_z'] + 2, cbDict['n_points_c'] + 2))
    YiPsi=np.zeros((cbDict['nYis']+1,cbDict['n_points_z'] + 2, cbDict['n_points_c'] + 2))
    Psi_compute=np.zeros((cbDict['nScalars']+1,cbDict['n_points_z'] -1, cbDict['n_points_c'] -1))
    YiPsi_compute=np.zeros((cbDict['nYis']+1,cbDict['n_points_z'] -1, cbDict['n_points_c'] -1))
    for k in range(2, cbDict['nScalars'] + 1):
        Psi[k,1:cbDict['n_points_z']+1,1:cbDict['n_points_c']+1]=Psicomputing(Src_vals[k],Src_vals[1],Src_vals[2],Src_vals[cbDict['nScalars']],c_space,z_space,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'],k)
    for k in range(1,cbDict['nYis']+1):
        YiPsi[k,1:cbDict['n_points_z']+1,1:cbDict['n_points_c']+1]=Psicomputing(Yi_vals[k],Src_vals[1],Src_vals[2],Src_vals[cbDict['nScalars']],c_space,z_space,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'],1)
    n_points_z=cbDict['n_points_z']
    n_points_c=cbDict['n_points_c']
    for k in range(2,cbDict['nScalars']+1):
        Psi_compute[k]=(1/4)*(Psi[k,1:n_points_z,1:n_points_c]+Psi[k,2:n_points_z+1,2:n_points_c+1]+Psi[k,1:n_points_z,2:n_points_c+1]+Psi[k,2:n_points_z+1,1:n_points_c])
    for k in range(1,cbDict['nYis']+1):
        YiPsi_compute[k]=(1/4)*(YiPsi[k,1:n_points_z,1:n_points_c]+YiPsi[k,2:n_points_z+1,2:n_points_c+1]+YiPsi[k,1:n_points_z,2:n_points_c+1]+YiPsi[k,2:n_points_z+1,1:n_points_c])
    if(cbDict['pdf_type']=="independent"):
         for iz in range(10,11):#1,cbDict['int_pts_z']+1
            index=int((cbDict['int_pts_gcz']+1)/2)
            paramlists.append((cbDict,iz,index,ih,z_space,c_space,Src_vals,Yi_vals,Psi_compute,YiPsi_compute))
         res=pool.map(pdf_multi,paramlists)
         pool.close()
         pool.join()
         print("time cost ",time.time()-start," s")
         return 0
    if(cbDict['pdf_type']=="frank"):
        for iz in range(10,11):#1,cbDict['int_pts_z']+1
            str_iz='%02d' % iz
            str_ih='%02d' % ih
            for igcz in range(1,cbDict['int_pts_gcz']+1):#cbDict['int_pts_gcz']+1
                paramlists.append((cbDict,iz,igcz,ih,z_space,c_space,Src_vals,Yi_vals,Psi_compute,YiPsi_compute))
        res=pool2.map(pdf_multi,paramlists)
        pool2.close()
        pool2.join()
        for iz in range(10,11):#1,cbDict['int_pts_z']+1
            str_iz='%02d' % iz
            str_ih='%02d' % ih
            strs=[]
            gcz_int=cbDict['gcz']
            for k in range(1,cbDict['int_pts_gcz']+1):
                rho=gcz_int[k]
                str_rho = '%.1f' % rho
                fp=open('unit'+str_iz+'_h'+str_ih+'_rho_'+str_rho+'.dat',"r")
                strs.append(fp.readlines())
                fp.close()
            with open('unit'+str_iz+'_h'+str_ih+'.dat',"a") as fl:
                for i in range(len(strs[0])):
                    for k in range(cbDict['int_pts_gcz']):
                        fl.write(strs[k][i])
            fl.close()
            for k in range(1,cbDict['int_pts_gcz']+1):
                rho=gcz_int[k]
                str_rho = '%.1f' % rho
                os.remove('unit'+str_iz+'_h'+str_ih+'_rho_'+str_rho+'.dat')
def assemble(cbDict):
  
  # read integrated units
  d2Yeq_table=cbDict['d2Yeq_table']
  #
  for i in range(1,cbDict['n_points_h']+1):
    #
    print('Reading unit' + '%02d' % 1 + '_h' + '%02d' % i + ' ... \n')
    M = np.loadtxt('unit01_h' + '%02d' % i + '.dat')
    #
    for j in range(2,cbDict['int_pts_z']+1):#cbDict['int_pts_z']+1
      #
      print('Reading unit' + '%02d' % j + '_h' + '%02d' % i + ' ... \n')
      #
      tmp = np.loadtxt('unit' + '%02d' % j + '_h' + '%02d' % i
                     + '.dat')
      #
      M = np.insert(tmp,0,M,axis=0)

  # remove unwanted columns - h(12),qdot(13),Yc_max(14)
    if(cbDict['scaled_PV']):
        rm_list = [12,13,14]
    else:
        rm_list = [12,13]
    MM = np.delete(M,rm_list,axis=1)

  # # separate scalars and Yis
  # if cbDict['nYis'] > 0:
  #   ind = -cbDict['nYis']
  #   MS = MM[:,:ind]
  #   YM = M[:,ind:]

  # write assembled table
  fln = cbDict['output_fln']
  print('Writing assembled table ...')
  with open(fln,'a') as strfile:
    #
    strfile.write(str(cbDict['int_pts_z']) + '\n')
    np.savetxt(strfile,cbDict['z'][1:],fmt='%.5E',delimiter='\t')
    #
    strfile.write(str(cbDict['int_pts_c']) + '\n')
    np.savetxt(strfile,cbDict['c'][1:],fmt='%.5E',delimiter='\t')
    #
    strfile.write(str(cbDict['int_pts_gz']) + '\n')
    np.savetxt(strfile,cbDict['gz'][1:],fmt='%.5E',delimiter='\t')
    #
    strfile.write(str(cbDict['int_pts_gc']) + '\n')
    np.savetxt(strfile,cbDict['gc'][1:],fmt='%.5E',delimiter='\t')
    #
    strfile.write(str(cbDict['int_pts_gcz']) + '\n')
    np.savetxt(strfile,cbDict['gcz'][1:],fmt='%.5E',delimiter='\t')
    #
    strfile.write(str(MM.shape[1]-5-cbDict['nYis']) + '\t' +
                  str(cbDict['nYis']) + '\n')
    np.savetxt(strfile,MM[:,5:],fmt='%.5E',delimiter='\t')
    #
    if(cbDict['scaled_PV']):
      np.savetxt(strfile,d2Yeq_table,fmt='%.5E')
    strfile.close()
    # for iz in range(1,cbDict['int_pts_z']+1):#1,cbDict['int_pts_z']+1
    #     str_iz='%02d' % iz
    #     ih=cbDict['n_points_h']
    #     str_ih='%02d' % ih
    #     os.remove('unit'+str_iz+'_h'+str_ih+'.dat')

  # #append Yis to the table
  # if cbDict['nYis'] > 0:
  #   #
  #   with open(fln,'a') as strfile:
  #     #
  #     strfile.write(str(cbDict['nYis']) + '\n')
  #     #
  #     np.savetxt(strfile,YM,fmt='%.5E',delimiter='\t')
  #     #
  #   strfile.close()


  print('\nDone.')