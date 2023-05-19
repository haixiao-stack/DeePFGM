import numpy as np
import time
import multiprocessing as mp
from scipy.stats import beta
import scipy.io as sio
from math import exp
import os
def c_dYdccomputing(sc_vals_int, sc_vals_int1, sc_vals_int2, sc_vals_intScalars, space, space_diff, space_average, mean,
                    cdf001, n_points_c, index):
    dsdc = np.zeros(n_points_c + 2)
    Yc_0 = np.zeros(n_points_c + 2)
    Yc_1 = np.zeros(n_points_c + 2)
    if (index < 2 or index > 4): dsdc[1:n_points_c] = (sc_vals_int[2:n_points_c + 1] - sc_vals_int[
                                                                                       1:n_points_c]) / space_diff[
                                                                                                        1:n_points_c]
    if (index == 2): dsdc[1:n_points_c] = (sc_vals_int2[2:n_points_c + 1] / sc_vals_int1[
                                                                            2:n_points_c + 1] - sc_vals_int2[
                                                                                                1:n_points_c] / sc_vals_int1[
                                                                                                                1:n_points_c]) / space_diff[
                                                                                                                                 1:n_points_c]
    if (index == 3):
        Yc_0[1:n_points_c] = space[1:n_points_c] * sc_vals_intScalars[1:n_points_c]
        Yc_1[2:n_points_c + 1] = space[2:n_points_c + 1] * sc_vals_intScalars[2:n_points_c + 1]
        dsdc[1:n_points_c] = (np.multiply(Yc_1[2:n_points_c + 1], sc_vals_int2[2:n_points_c + 1] / sc_vals_int1[
                                                                                                   2:n_points_c + 1]) - np.multiply(
            Yc_0[1:n_points_c], sc_vals_int2[1:n_points_c] / sc_vals_int1[1:n_points_c])) / space_diff[1:n_points_c]
    if (index == 4):
        dsdc[1:n_points_c] = (mean * sc_vals_int2[2:n_points_c + 1] / sc_vals_int1[
                                                                      2:n_points_c + 1] - mean * sc_vals_int2[
                                                                                                 1:n_points_c] / sc_vals_int1[
                                                                                                                 1:n_points_c]) / space_diff[
                                                                                                                                  1:n_points_c]
    dsdc[n_points_c] = dsdc[n_points_c - 1]
    dsdc[n_points_c + 1] = dsdc[1]
    y_int = 0
    y_int = y_int - np.sum(0.5 * np.multiply((np.multiply(dsdc[1:n_points_c - 1],
                                                          cdf001[1:n_points_c - 1]) + np.multiply(dsdc[2:n_points_c],
                                                                                                  cdf001[
                                                                                                  2:n_points_c])),
                                             (space_average[2:n_points_c] - space_average[1:n_points_c - 1])))
    y_int = y_int - dsdc[n_points_c] * cdf001[n_points_c] * (space[n_points_c] - space[n_points_c - 1]) / 2.0 - dsdc[
        n_points_c + 1] * cdf001[1] * (space[2] - space[1]) / 2.0 + sc_vals_int[n_points_c]
    return y_int
def z_dYdccomputing(sc_vals_int, sc_vals_int1, sc_vals_int2, sc_vals_intScalars, space, space_diff, space_average, mean,
                    cdf001, n_points_z, index):
    dsdc = np.zeros(n_points_z + 2)
    Yc_0 = np.zeros(n_points_z + 2)
    Yc_1 = np.zeros(n_points_z + 2)
    if (index < 2 or index > 4): dsdc[1:n_points_z] = (sc_vals_int[2:n_points_z + 1] - sc_vals_int[
                                                                                       1:n_points_z]) / space_diff[
                                                                                                        1:n_points_z]
    if (index == 2): dsdc[1:n_points_z] = (sc_vals_int2[2:n_points_z + 1] / sc_vals_int1[
                                                                            2:n_points_z + 1] - sc_vals_int2[
                                                                                                1:n_points_z] / sc_vals_int1[
                                                                                                                1:n_points_z]) / space_diff[
                                                                                                                                 1:n_points_z]
    if (index == 3):
        Yc_0[1:n_points_z] = mean * sc_vals_intScalars[1:n_points_z]
        Yc_1[2:n_points_z + 1] = mean * sc_vals_intScalars[2:n_points_z + 1]
        dsdc[1:n_points_z] = (np.multiply(Yc_1[2:n_points_z + 1], sc_vals_int2[2:n_points_z + 1] / sc_vals_int1[
                                                                                                   2:n_points_z + 1]) - np.multiply(
            Yc_0[1:n_points_z], sc_vals_int2[1:n_points_z] / sc_vals_int1[1:n_points_z])) / space_diff[1:n_points_z]
    if (index == 4):
        dsdc[1:n_points_z] = (space[2:n_points_z + 1] * sc_vals_int2[2:n_points_z + 1] / sc_vals_int1[
                                                                                         2:n_points_z + 1] - space[
                                                                                                             1:n_points_z] * sc_vals_int2[
                                                                                                                             1:n_points_z] / sc_vals_int1[
                                                                                                                                             1:n_points_z]) / space_diff[
                                                                                                                                                              1:n_points_z]
    dsdc[n_points_z] = dsdc[n_points_z - 1]
    dsdc[n_points_z + 1] = dsdc[1]
    y_int = 0
    y_int = y_int - np.sum(0.5 * np.multiply((np.multiply(dsdc[1:n_points_z - 1],
                                                          cdf001[1:n_points_z - 1]) + np.multiply(dsdc[2:n_points_z],
                                                                                                  cdf001[
                                                                                                  2:n_points_z])),
                                             (space_average[2:n_points_z] - space_average[1:n_points_z - 1])))
    y_int = y_int - dsdc[n_points_z] * cdf001[n_points_z] * (space[n_points_z] - space[n_points_z - 1]) / 2.0 - dsdc[
        n_points_z + 1] * cdf001[1] * (space[2] - space[1]) / 2.0 + sc_vals_int[n_points_z]
    return y_int
def c_cdfFunc(space_average, alpha_c, beta_c, n_points_c):
    cdf001 = beta.cdf(space_average, alpha_c, beta_c)
    cdf001[n_points_c] = 1
    return cdf001
def z_cdfFunc(space_average, alpha_z, beta_z, n_points_z):
    cdf001 = beta.cdf(space_average, alpha_z, beta_z)
    cdf001[n_points_z] = 1
    return cdf001
def processdata(data,k,x,y):
    data_new=[]
    for i in range(0,k):
        data_new.append(data[:,:,i])
    data_new=np.array(data_new)
    return data_new
def intfac(x,xarray,loc_low):
    if(x<xarray[loc_low]): return 0
    if (x > xarray[loc_low+1]): return 1
    return (x-xarray[loc_low])/(xarray[loc_low+1]-xarray[loc_low])
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
# cbeta function
def cbeta(mean, g_var, space, space_average, space_diff, z_mean, z_space, sc_vals, Yi_vals, n_points_z, n_points_c,
          nScalars, nYis):
    loc = locate(z_space, n_points_z, z_mean)
    fac = intfac(z_mean, z_space, loc)
    sc_vals_int = np.zeros((nScalars + 1, n_points_c + 1))
    Yi_vals_int = np.zeros((nYis + 1, n_points_c + 1))
    y_int = np.zeros(nScalars + 1)
    Yi_int = np.zeros(nYis + 1)
    alpha_c = mean * ((1 / g_var) - 1)
    beta_c = (1 - mean) * ((1 / g_var) - 1)
    sc_vals_int[1:nScalars + 1, 1:n_points_c + 1] = fac * sc_vals[1:nScalars + 1, loc + 1, 1:n_points_c + 1] + (
                1 - fac) * sc_vals[1:nScalars + 1, loc, 1:n_points_c + 1]
    Yi_vals_int[1:nYis + 1, 1:n_points_c + 1] = fac * Yi_vals[1:nYis + 1, loc + 1, 1:n_points_c + 1] + (
                1 - fac) * Yi_vals[1:nYis + 1, loc, 1:n_points_c + 1]
    cdf001 = c_cdfFunc(space_average, alpha_c, beta_c, n_points_c)
    sc_vals_int1 = sc_vals_int[1]
    sc_vals_int2 = sc_vals_int[2]
    sc_vals_intScalars = sc_vals_int[nScalars]
    for j in range(1, nScalars + 1):
        y_int[j] = c_dYdccomputing(sc_vals_int[j], sc_vals_int1, sc_vals_int2, sc_vals_intScalars, space, space_diff,
                                   space_average, z_mean, cdf001, n_points_c, j)
    for j in range(1, nYis + 1):
        Yi_int[j] = c_dYdccomputing(Yi_vals_int[j], sc_vals_int1, sc_vals_int2, sc_vals_intScalars, space, space_diff,
                                    space_average, z_mean, cdf001, n_points_c, 1)
    y_int = np.where(abs(y_int) > 1e-20, y_int, 0)
    Yi_int = np.where(Yi_int > 1e-20, Yi_int, 0)
    return y_int, Yi_int
def zbeta(mean, g_var, space, space_average, space_diff, c_mean, c_space, sc_vals, Yi_vals, n_points_z, n_points_c,
          nScalars, nYis):
    loc = locate(c_space, n_points_c, c_mean)
    fac = intfac(c_mean, c_space, loc)
    sc_vals_int = np.zeros((nScalars + 1, n_points_z + 1))
    Yi_vals_int = np.zeros((nYis + 1, n_points_z + 1))
    y_int = np.zeros(nScalars + 1)
    Yi_int = np.zeros(nYis + 1)
    alpha_z = mean * ((1 / g_var) - 1)
    beta_z = (1 - mean) * ((1 / g_var) - 1)
    sc_vals_int[1:nScalars + 1, 1:n_points_z + 1] = fac * sc_vals[1:nScalars + 1, 1:n_points_z + 1, loc + 1] + (
                1 - fac) * sc_vals[1:nScalars + 1, 1:n_points_z + 1, loc]
    Yi_vals_int[1:nYis + 1, 1:n_points_z + 1] = fac * Yi_vals[1:nYis + 1, 1:n_points_z + 1, loc + 1] + (
                1 - fac) * Yi_vals[1:nYis + 1, 1:n_points_z + 1, loc]
    cdf001 = z_cdfFunc(space_average, alpha_z, beta_z, n_points_z)
    sc_vals_int1 = sc_vals_int[1]
    sc_vals_int2 = sc_vals_int[2]
    sc_vals_intScalars = sc_vals_int[nScalars]
    for j in range(1, nScalars + 1):
        y_int[j] = z_dYdccomputing(sc_vals_int[j], sc_vals_int1, sc_vals_int2, sc_vals_intScalars, space, space_diff,
                                   space_average, c_mean, cdf001, n_points_z, j)
    for j in range(1, nYis + 1):
        Yi_int[j] = z_dYdccomputing(Yi_vals_int[j], sc_vals_int1, sc_vals_int2, sc_vals_intScalars, space, space_diff,
                                    space_average, c_mean, cdf001, n_points_z, 1)
    y_int = np.where(abs(y_int) > 1e-20, y_int, 0)
    Yi_int = np.where(Yi_int > 1e-20, Yi_int, 0)
    return y_int, Yi_int
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
def readcopula(filename):
    roadef_info = sio.loadmat(filename)
    prob = roadef_info['y'][0]
    return prob
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
def Psicomputing(sc_vals,sc_vals1,sc_vals2,sc_valsScalars,c_space,z_space,n_points_z,n_points_c,nScalars,nYis,k):
    z_space_use = np.reshape(np.repeat(z_space, n_points_c + 1), (n_points_z + 1, n_points_c + 1))
    c_space_use = np.reshape(np.tile(c_space, n_points_z + 1), (n_points_z + 1, n_points_c + 1))
    if(k<2 or k>4): return sc_vals[1:n_points_z+1,1:n_points_c + 1]
    if(k==2): return sc_vals[1:n_points_z+1,1:n_points_c + 1]/ sc_vals1[1:n_points_z+1,1:n_points_c + 1]
    if(k==3): return np.multiply(np.multiply(np.multiply(sc_vals2[1:n_points_z+1,1:n_points_c + 1], 1/ sc_vals1[1:n_points_z+1,1:n_points_c + 1]), c_space_use[1:n_points_z+1,1:n_points_c + 1]), sc_valsScalars[1:n_points_z+1,1:n_points_c + 1])
    if(k==4): return np.multiply(np.multiply(sc_vals2[1:n_points_z+1,1:n_points_c + 1], 1 / sc_vals1[1:n_points_z+1,1:n_points_c + 1]), z_space_use[1:n_points_z+1,1:n_points_c + 1])
def CDF_ind(z_space, c_space, z_space_average, c_space_average, alpha_z, beta_z, alpha_c, beta_c, n_points_c,
            n_points_z):
    CDF_C = np.zeros(n_points_c + 2)
    CDF_Z = np.zeros(n_points_z + 2)
    CDF_Z[1:n_points_z + 1] = beta.cdf(z_space_average[1:n_points_z + 1], alpha_z, beta_z)
    CDF_C[1:n_points_c + 1] = beta.cdf(c_space_average[1:n_points_c + 1], alpha_c, beta_c)
    j = n_points_c
    CDF_C[j] = beta.cdf((c_space[j - 1] + 3 * c_space[j]) / 4.0, alpha_c, beta_c)
    j = n_points_c + 1
    CDF_C[j] = beta.cdf((3 * c_space[1] + 2 * c_space[2]) / 4.0, alpha_c, beta_c)
    i = n_points_z
    CDF_Z[i] = beta.cdf((z_space[i - 1] + 3 * z_space[i]) / 4.0, alpha_z, beta_z)
    i = n_points_z + 1
    CDF_Z[i] = beta.cdf((3 * z_space[1] + z_space[2]) / 4.0, alpha_z, beta_z)
    return CDF_C, CDF_Z
def CDF_copula(z_space, c_space,alpha_z, beta_z, alpha_c, beta_c, n_points_c,n_points_z,type,rho,parameters):
    CDF_C = np.zeros(n_points_c + 2)
    CDF_Z = np.zeros(n_points_z + 2)
    # print("zspace",z_space)
    # print("cspace",c_space)
    CDF_Z[1:n_points_z + 1] = beta.cdf(z_space[1:n_points_z + 1], alpha_z, beta_z)
    CDF_C[1:n_points_c + 1] = beta.cdf(c_space[1:n_points_c + 1], alpha_c, beta_c)
    # j = n_points_c
    # CDF_C[j] = beta.cdf((c_space[j - 1] + 3 * c_space[j]) / 4.0, alpha_c, beta_c)
    # i = n_points_z
    # CDF_Z[i] = beta.cdf((z_space[i - 1] + 3 * z_space[i]) / 4.0, alpha_z, beta_z)
    if (rho == 0): type = "independent"
    # print(CDF_C)
    X,Y = np.meshgrid(CDF_C, CDF_Z)
    # print("X",X)
    # print("Y",Y)
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
def dPsicomputing(Psi, dPsidc, c_space, c_space_average_use, z_space, z_space_average, z_space_diff, CDF_C, CDF_Z,n_points_z, n_points_c, bias):
    Q_int = np.zeros(n_points_z + 2)
    dQdz = np.zeros(n_points_z + 2)
    # time4=time.time()
    CDF_C_use = np.reshape(np.tile(CDF_C, n_points_z + 1), (n_points_z + 1, n_points_c + 2))
    Q_int[1:n_points_z + 1] = Q_int[1:n_points_z + 1] - np.sum(0.5 * np.multiply((np.multiply(dPsidc[1:n_points_z + 1, 1:n_points_c - 1], CDF_C_use[1:n_points_z + 1, 1:n_points_c - 1]) + np.multiply(dPsidc[1:n_points_z + 1, 2:n_points_c], CDF_C_use[1:n_points_z + 1, 2:n_points_c])), (c_space_average_use[1:n_points_z + 1,2:n_points_c] - c_space_average_use[1:n_points_z + 1,1:n_points_c - 1])),axis=1)
    Q_int[1:n_points_z + 1] = Q_int[1:n_points_z + 1] - np.multiply(np.multiply(dPsidc[1:n_points_z + 1, n_points_c], np.repeat(CDF_C[n_points_c], n_points_z)),np.repeat((c_space[n_points_c] - c_space[n_points_c - 1]) / 2.0, n_points_z)) - np.multiply(np.multiply(dPsidc[1:n_points_z + 1, n_points_c + 1], np.repeat(CDF_C[n_points_c + 1], n_points_z)),np.repeat((c_space[2] - c_space[1]) / 2.0, n_points_z)) + Psi[1:n_points_z + 1,n_points_c]
    dQdz[1:n_points_z] = (Q_int[2:n_points_z + 1] - Q_int[1:n_points_z] + bias) / z_space_diff[1:n_points_z]
    dQdz[n_points_z] = dQdz[n_points_z - 1]
    dQdz[n_points_z + 1] = dQdz[1]
    yint = -np.sum(
        0.5 * (dQdz[1:n_points_z - 1] * CDF_Z[1:n_points_z - 1] + dQdz[2:n_points_z] * CDF_Z[2:n_points_z]) * (
                    z_space_average[2:n_points_z] - z_space_average[1:n_points_z - 1]))
    yint = yint + dQdz[n_points_z] * CDF_Z[n_points_z] * (z_space[n_points_z] - z_space[n_points_z - 1]) / 2.0 + dQdz[
        n_points_z + 1] * CDF_Z[n_points_z + 1] * (z_space[2] - z_space[1]) / 2.0 + Q_int[n_points_z]
    # print("yint",yint)
    # time7=time.time()
    # print("Calculate4 time",time7-time6)
    return yint
def int_point_ind(z_mean, c_mean, c_var, z_var,z_space, c_space, z_space_average, c_space_average, z_space_diff,
              c_space_diff, c_space_average_use, Psi, YiPsi, dPsidc, dYiPsidc, n_points_z, n_points_c, nScalars, nYis):
    y_int = np.zeros(nScalars + 1)
    Yi_int = np.zeros(nYis + 1)
    alpha_z = z_mean * (((z_mean * (1 - z_mean)) / z_var) - 1)
    alpha_c = c_mean * (((c_mean * (1 - c_mean)) / c_var) - 1)
    beta_z = (1 - z_mean) * (((z_mean * (1 - z_mean)) / z_var) - 1)
    beta_c = (1 - c_mean) * (((c_mean * (1 - c_mean)) / c_var) - 1)
    CDF_C, CDF_Z= CDF_ind(z_space, c_space, z_space_average, c_space_average, alpha_z, beta_z, alpha_c, beta_c,
                                n_points_c, n_points_z)
    for k in range(2, nScalars + 1):
        # print(k)
        if (k < 5):
            y_int[k] = dPsicomputing(Psi[k], dPsidc[k], c_space, c_space_average_use, z_space, z_space_average,
                                     z_space_diff, CDF_C, CDF_Z, n_points_z, n_points_c, 1e-15)
        else:
            y_int[k] = dPsicomputing(Psi[k], dPsidc[k], c_space, c_space_average_use, z_space, z_space_average,
                                     z_space_diff, CDF_C, CDF_Z, n_points_z, n_points_c, 0)
    for k in range(1, nYis + 1):
        Yi_int[k] = dPsicomputing(YiPsi[k], dYiPsidc[k], c_space, c_space_average_use, z_space, z_space_average,
                                  z_space_diff, CDF_C, CDF_Z, n_points_z, n_points_c, 0)
    # time4=time.time()
    # print("dPsi time",time4-time3)
    y_int = np.where(abs(y_int) > 1e-20, y_int, 0)
    Yi_int = np.where(Yi_int > 1e-20, Yi_int, 0)
    # print("yint",y_int,Yi_int)
    return y_int, Yi_int
def int_point_copula(z_mean,c_mean,c_var,z_var,rho,z_space,c_space,Psi_compute,YiPsi_compute,n_points_z,n_points_c,nScalars,nYis,type,parameters):
    y_int=np.zeros(nScalars+1)
    Yi_int=np.zeros(nYis+1)
    alpha_z = z_mean * (((z_mean * (1-z_mean)) / z_var) - 1)
    alpha_c = c_mean * (((c_mean * (1-c_mean)) / c_var) - 1)
    beta_z = (1-z_mean) * (((z_mean * (1-z_mean)) / z_var) - 1)
    beta_c = (1-c_mean) * (((c_mean * (1-c_mean)) / c_var) - 1)
    CDF_C, CDF_Z, CDF_multi = CDF_copula(z_space, c_space,alpha_z, beta_z, alpha_c, beta_c, n_points_c, n_points_z,type,rho,parameters)
    CDF_multi_compute=np.zeros((n_points_z-1,n_points_c-1))
    CDF_multi_compute=(CDF_multi[1:n_points_z,1:n_points_c]+CDF_multi[2:n_points_z+1,2:n_points_c+1]-CDF_multi[1:n_points_z,2:n_points_c+1]-CDF_multi[2:n_points_z+1,1:n_points_c])
    # print(CDF_multi)
    for k in range(5,10):
        y_int[k] = np.sum(np.multiply(Psi_compute[k],CDF_multi_compute))
    # for k in range(1,nYis+1):
    #     Yi_int[k] = np.sum(np.multiply(YiPsi_compute[k],CDF_multi_compute))
    y_int = np.where(abs(y_int) > 1e-20, y_int, 0)
    Yi_int = np.where(Yi_int > 1e-20, Yi_int, 0)
    return y_int,Yi_int
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
    dPsidc = np.zeros((cbDict['nScalars'] + 1, cbDict['n_points_z'] + 2, cbDict['n_points_c'] + 2))
    dYiPsidc = np.zeros((cbDict['nYis'] + 1, cbDict['n_points_z'] + 2, cbDict['n_points_c'] + 2))
    c_space_diff = computedata(c_space, "diff", cbDict['n_points_c'])
    c_space_diff_use = np.reshape(np.tile(c_space_diff, cbDict['n_points_z'] + 1),(cbDict['n_points_z'] + 1, cbDict['n_points_c'] + 1))
    for k in range(2, cbDict['nScalars'] + 1):
        Psi[k, 1:cbDict['n_points_z'] + 1, 1:cbDict['n_points_c'] + 1] = Psicomputing(Src_vals[k], Src_vals[1],Src_vals[2],Src_vals[cbDict['nScalars']],c_space, z_space,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'], k)
    for k in range(1, cbDict['nYis'] + 1):
        YiPsi[k, 1:cbDict['n_points_z'] + 1, 1:cbDict['n_points_c'] + 1] = Psicomputing(Yi_vals[k], Src_vals[1],Src_vals[2],Src_vals[cbDict['nScalars']],c_space, z_space,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'], 1)
    for k in range(2, cbDict['nScalars'] + 1):
        if (k < 5):
            dPsidc[k, 1:cbDict['n_points_z'] + 1, 1:cbDict['n_points_c']] = (Psi[k, 1:cbDict['n_points_z'] + 1,2:cbDict['n_points_c'] + 1] - Psi[k,1:cbDict['n_points_z'] + 1,1:cbDict['n_points_c']] + 1e-15) / c_space_diff_use[1:cbDict['n_points_z'] + 1,1:cbDict['n_points_c']]
        else:
            dPsidc[k, 1:cbDict['n_points_z'] + 1, 1:cbDict['n_points_c']] = (Psi[k, 1:cbDict['n_points_z'] + 1,2:cbDict['n_points_c'] + 1] - Psi[k,1:cbDict['n_points_z'] + 1,1:cbDict['n_points_c']]) / c_space_diff_use[1:cbDict['n_points_z'] + 1,1:cbDict['n_points_c']]
        dPsidc[k, 1:cbDict['n_points_z'] + 1, cbDict['n_points_c']] = dPsidc[k, 1:cbDict['n_points_z'] + 1,cbDict['n_points_c'] - 1]
        dPsidc[k, 1:cbDict['n_points_z'] + 1, cbDict['n_points_c'] + 1] = dPsidc[k, 1:cbDict['n_points_z'] + 1, 1]
    for k in range(1, cbDict['nYis'] + 1):
        dYiPsidc[k, 1:cbDict['n_points_z'] + 1, 1:cbDict['n_points_c']] = (YiPsi[k, 1:cbDict['n_points_z'] + 1,2:cbDict['n_points_c'] + 1] - YiPsi[k,1:cbDict['n_points_z'] + 1,1:cbDict['n_points_c']]) / c_space_diff_use[1:cbDict['n_points_z'] + 1,1:cbDict['n_points_c']]
        dYiPsidc[k, 1:cbDict['n_points_z'] + 1, cbDict['n_points_c']] = dYiPsidc[k, 1:cbDict['n_points_z'] + 1,cbDict['n_points_c'] - 1]
        dYiPsidc[k, 1:cbDict['n_points_z'] + 1, cbDict['n_points_c'] + 1] = dYiPsidc[k, 1:cbDict['n_points_z'] + 1, 1]
    n_points_z=cbDict['n_points_z']
    n_points_c=cbDict['n_points_c']
    for k in range(2,cbDict['nScalars']+1):
        Psi_compute[k]=(1/4)*(Psi[k,1:n_points_z,1:n_points_c]+Psi[k,2:n_points_z+1,2:n_points_c+1]+Psi[k,1:n_points_z,2:n_points_c+1]+Psi[k,2:n_points_z+1,1:n_points_c])
    for k in range(1,cbDict['nYis']+1):
        YiPsi_compute[k]=(1/4)*(YiPsi[k,1:n_points_z,1:n_points_c]+YiPsi[k,2:n_points_z+1,2:n_points_c+1]+YiPsi[k,1:n_points_z,2:n_points_c+1]+YiPsi[k,2:n_points_z+1,1:n_points_c])
    # if(cbDict['pdf_type']=="independent"):
    for iz in range(1,cbDict['int_pts_z']+1):#1,cbDict['int_pts_z']+1
            paramlists.append((cbDict,iz,ih,z_space,c_space,Src_vals,Yi_vals,Psi_compute,YiPsi_compute,Psi,YiPsi,dPsidc,dYiPsidc))
    res=pool.map(pdf_multi,paramlists)
    pool.close()
    pool.join()
    print("time cost ",time.time()-start," s")
    return 0
    # if(cbDict['pdf_type']=="frank"):
    #     for iz in range(10,11):#1,cbDict['int_pts_z']+1
    #         str_iz='%02d' % iz
    #         str_ih='%02d' % ih
    #         for igcz in range(1,cbDict['int_pts_gcz']+1):#cbDict['int_pts_gcz']+1
    #             paramlists.append((cbDict,iz,ih,z_space,c_space,Src_vals,Yi_vals,Psi_compute,YiPsi_compute))
    #     res=pool2.map(pdf_multi,paramlists)
    #     pool2.close()
    #     pool2.join()
    #     for iz in range(10,11):#1,cbDict['int_pts_z']+1
    #         str_iz='%02d' % iz
    #         str_ih='%02d' % ih
    #         strs=[]
    #         gcz_int=cbDict['gcz']
    #         for k in range(1,cbDict['int_pts_gcz']+1):
    #             rho=gcz_int[k]
    #             str_rho = '%.1f' % rho
    #             fp=open('unit'+str_iz+'_h'+str_ih+'_rho_'+str_rho+'.dat',"r")
    #             strs.append(fp.readlines())
    #             fp.close()
    #         with open('unit'+str_iz+'_h'+str_ih+'.dat',"a") as fl:
    #             for i in range(len(strs[0])):
    #                 for k in range(cbDict['int_pts_gcz']):
    #                     fl.write(strs[k][i])
    #         fl.close()
    #         for k in range(1,cbDict['int_pts_gcz']+1):
    #             rho=gcz_int[k]
    #             str_rho = '%.1f' % rho
    #             os.remove('unit'+str_iz+'_h'+str_ih+'_rho_'+str_rho+'.dat')
def pdf_multi(item):
    cbDict,iz,ih,z_space,c_space,Src_vals,Yi_vals,Psi_compute,YiPsi_compute,Psi,YiPsi,dPsidc,dYiPsidc=item
    parameters = {}
    parameters['frank'] = readcopula("./frankparameters.mat")
    parameters['placket'] = readcopula("./placketparameters.mat")
    integrate(cbDict,z_space,c_space,Src_vals,Yi_vals,iz,ih,parameters,Psi_compute,YiPsi_compute,Psi,YiPsi,dPsidc,dYiPsidc)
def integrate(cbDict,z_space,c_space,Src_vals,Yi_vals,iz,ih,parameters,Psi_compute,YiPsi_compute,Psi,YiPsi,dPsidc,dYiPsidc):
        start = time.time()
        yint = np.zeros(cbDict['nScalars'] + 1)
        Yi_int = np.zeros(cbDict['nYis'] + 1)
        p=0
        str_iz='%02d' % iz
        str_ih='%02d' % ih
        z_int=cbDict['z']
        c_int=cbDict['c']
        gz_int=cbDict['gz']
        gc_int=cbDict['gc']
        gcz_int=cbDict['gcz']
        #   rho=gcz_int[igcz]
        #   str_rho = '%.1f' % rho
        z_space_average = computedata(z_space, "average", cbDict['n_points_z'])
        z_space_diff = computedata(z_space, "diff", cbDict['n_points_z'])
        c_space_average = computedata(c_space, "average", cbDict['n_points_c'])
        c_space_diff = computedata(c_space, "diff", cbDict['n_points_c'])
        c_space_average_use = np.reshape(np.tile(c_space_average, cbDict['n_points_z'] + 1),
                                        (cbDict['n_points_z'] + 1, cbDict['n_points_c'] + 1))
        #   print(cbDict["pdf_type"])
        #   if(cbDict["pdf_type"]=="independent"):
        f = open('./canteraData/unit'+str_iz+'_h'+str_ih+'.dat', "w")
    #   else:
    #     f = open('unit'+str_iz+'_h'+str_ih+'_rho_'+str_rho+'.dat', "w")
        p=0
        z_loc = locate(z_space, cbDict['n_points_z'], z_int[iz])
        for ic in range(1,cbDict['int_pts_c']+1):#1,cbDict['int_pts_c']+1
            for igz in range(1,cbDict['int_pts_gz']+1):#1,cbDict['int_pts_gz']+1
                for igc in range(1,cbDict['int_pts_gc']+1):#1,cbDict['int_pts_gc']+1
                    p=p+1
                    #print("computing unit"+str_iz+" case ",p," ",iz," ",ic," ",igz," ",igc," ",0)#显示计算进程，嫌烦这行可以删除
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
                            yint,Yi_int=int_point_ind(z_int[iz],c_int[ic],c_var,z_var,z_space,c_space,z_space_average,c_space_average,z_space_diff,c_space_diff,c_space_average_use, Psi, YiPsi, dPsidc, dYiPsidc,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'])#cbDict['pdf_type']
                    if(cbDict["pdf_type"]=="independent"):
                        data=[z_int[iz],c_int[ic],gz_int[igz],gc_int[igc],0]+[x for x in yint[2:(cbDict['nScalars']+1)]]+[x for x in Yi_int[1:]]
                        for i in range(0,len(data)):
                            f.write(str('%0.5E' % data[i]))
                            f.write(" ")
                        f.write("\n")
                    if(cbDict["pdf_type"]=="frank"):
                        # yint_use,Yi_int_use=yint,Yi_int
                        for igcz in range(1,cbDict['int_pts_gcz']+1):
                            if(igcz==((cbDict['int_pts_gcz']/2)+1)):
                                data=[z_int[iz],c_int[ic],gz_int[igz],gc_int[igc],gcz_int[igcz]]+[x for x in yint[2:(cbDict['nScalars']+1)]]+[x for x in Yi_int[1:]]
                            else:
                                yint_use,Yi_int_use=yint,Yi_int
                                if(igz!=1 and igc!=1 and iz>1 and iz<cbDict['int_pts_z'] and ic>1 and ic<cbDict['int_pts_c']):
                                    c_var=gc_int[igc]*(c_int[ic]*(1.0-c_int[ic]))
                                    z_var=gz_int[igz]*(z_int[iz]*(1.0-z_int[iz]))
                                    yint_copula,Yi_int_copula=int_point_copula(z_int[iz],c_int[ic],c_var,z_var,gcz_int[igcz],z_space,c_space,Psi_compute,YiPsi_compute,cbDict['n_points_z'],cbDict['n_points_c'],cbDict['nScalars'],cbDict['nYis'],cbDict['pdf_type'],parameters)
                                    yint_use[5:10]=yint_copula[5:10]
                                data=[z_int[iz],c_int[ic],gz_int[igz],gc_int[igc],gcz_int[igcz]]+[x for x in yint_use[2:(cbDict['nScalars']+1)]]+[x for x in Yi_int_use[1:]]
                            for i in range(0,len(data)):
                                f.write(str('%0.5E' % data[i]))
                                f.write(" ")
                            f.write("\n")
        # print("Writing done ",'unit'+str_iz+'_h'+str_ih+'_rho_'+str_rho+'.dat')
        # print("写该文件总耗时"," ",time.time()-start,"s")
        f.close()