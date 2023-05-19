import ctypes
from ntpath import join
import cantera as ct
import numpy as np
import sys
import os
from multiprocessing import Pool,Array
# import time
# import zipfile,tarfile

def Sim(i, Z, phi, fuel_species, CASENAME, chemMech, transModel, nSpeMech, nVarCant, \
        p, Lx, stoich_O2, T_fuel, X_fuel, T_ox, X_ox, solFln):
    #

    reactants = {fuel_species: phi[i] / stoich_O2, 'O2': 1.0, 'N2': 3.76}

    ## Load chemical mechanism
    gas = ct.Solution(chemMech)

    # Stream A (air)
    A = ct.Quantity(gas, constant='HP')
    A.TPX = T_ox, p, X_ox # Define oxidiser

    # Stream B (methane)
    B = ct.Quantity(gas, constant='HP')
    B.TPX = T_fuel, p, X_fuel # Define fuel

    # Set the molar flow rates corresponding to stoichiometric reaction,
    # CH4 + 2 O2 -> CO2 + 2 H2O
    A.moles = 1
    nO2 = A.X[A.species_index('O2')]
    B.moles = nO2 / stoich_O2 * phi[i]

    # Compute the mixed state
    M = A + B

    # unburned gas temperature [K]
    Tin = M.T
    print('Flamelet No. ',i,'|',phi[i],Z[i],Tin,p,reactants)

    # set reactants state
    gas.TPX = Tin, p, reactants

    # Flame object
    f = ct.FreeFlame(gas,width=Lx)

    if transModel == 'Mix' or transModel == 'Multi':    
        #Mix：不同组分的分子输运参数平均后计算；Mixture：不同组分用自己的分子输运参数
        # Solve with the energy equation disabled
        f.energy_enabled = False
        f.transport_model = transModel
        f.solve(loglevel=1, refine_grid=False)

        # Solve with the energy equation enabled
        f.set_refine_criteria(ratio=3, slope=0.07, curve=0.14)
        f.energy_enabled = True
        f.solve(loglevel=1,auto=True)
        print('mixture-averaged flamespeed = {0:7f} m/s'.format(f.velocity[0]))   
        #为什么有无能量方程都求一遍？

    elif transModel == 'Multi':
        # Solve with multi-component transport properties
        f.transport_model = 'Multi'
        f.solve(loglevel=1, auto=True)
        print('multicomponent flamespeed = {0:7f} m/s'.format(f.velocity[0]))     
        #transModel == 'Multi'为什么还要再求一遍？

    else:
        sys.exit(' !! Error - incorrect transport model specified !!')
    
    sL.append(f.velocity[0])
    # sL.append(f.u[0]) # store flame speed in array sL

    # store useful data for future simulation
    iCO = gas.species_index('CO')
    iCO2 = gas.species_index('CO2')
    data = np.zeros((len(f.grid),nSpeMech+nVarCant))  

    # unscaled progress variable
    #data第0列：CO和CO2的质量分数之和，表示反应进度
    data[:,0] = f.Y[iCO] + f.Y[iCO2]      

    # Reaction rate of unscaled progress variable
    #data第1列：CO摩尔质量*CO净生成率+CO2摩尔质量*CO2净生成率，表示反应速率
    data[:,1] = f.gas.molecular_weights[iCO]*f.net_production_rates[iCO,:] \
        + f.gas.molecular_weights[iCO2]*f.net_production_rates[iCO2,:]    

    data[:,2] = f.T       #data第2列：温度
    data[:,3] = f.density_mass   #data第3列：密度
    data[:,4] = f.cp_mass    #data第4列：定压热容
    #data第5列：摩尔分数转置后与摩尔质量的点积，即质量
    data[:,5] = np.dot(np.transpose(f.X),f.gas.molecular_weights)   

    # formation enthalpy 生成焓
    for j in range(len(f.grid)):
        dumGas = ct.Solution(chemMech) # dummy working variable
        dumGas.TPY = 298.15,p,f.Y[:,j]
        data[j,6] = dumGas.enthalpy_mass   

    data[:,7] = f.viscosity/f.density   #data第7列：运动粘度=动力粘度/密度
    data[:,8] = f.enthalpy_mass         #data第8列：焓
    data[:,9] = f.heat_release_rate     #data第9列：释热率

    data[:,nVarCant:nSpeMech+nVarCant] = np.transpose(f.Y)   
            #data最后nSpeMech个列：组分质量分数矩阵的转置

    # save flamelet data
    fln = solFln + CASENAME + '_' + '{:03d}'.format(i) + '.csv'    
    np.savetxt(fln,data)                                           

    ###### Compute global flame properties #######
    ###### calculate flame thickness
    DT = np.gradient(f.T,f.grid)  #温度梯度（相对空间位置）
    dl = (f.T[-1]-f.T[0])/max(DT)   #火焰厚度

    phi_tabi = np.frombuffer(phi_tab[i],dtype=ctypes.c_double)

    phi_tabi[0] = phi[i]                      #equivalence ratio
    phi_tabi[1] = Z[i]                        #mixture fraction
    phi_tabi[2] = len(f.grid)                 #number of grid points
    phi_tabi[3] = f.velocity[0]                      #flame speed
    phi_tabi[4] = dl                          #flame thickness
    phi_tabi[5] = (f.T[-1]-f.T[0])/f.T[0]     #heat release parameter

    c = (f.Y[iCO,:] + f.Y[iCO2,:]) / max(f.Y[iCO,:] + f.Y[iCO2,:])  #相对反应进度
    alpha = f.thermal_conductivity/f.density/f.cp_mass                 
    #热导率/密度/定压比热=单位面积下由于热传导引起的温度变化
    Dc = np.gradient(c,f.grid)   #反应进度的空间梯度
    Nc = alpha*Dc*Dc                                                   
    PDF_c = Dc*f.viscosity/f.density/f.velocity[0]                            
    integ_1 = np.trapz(f.density*Nc*np.gradient(f.velocity,f.grid)*PDF_c,c)
    integ_2 = np.trapz(f.density*Nc*PDF_c,c)
    phi_tabi[6] = dl/f.velocity[0] *integ_1/integ_2/phi_tabi[5]   #KcStar   

    ###### calculate integral of cp in T space from 298.15 to Tin
    gasCP = ct.Solution(chemMech)
    gasCP.TPX = 298.15,p,reactants
    cp_0 = gasCP.cp_mass
    if abs(Tin - 298.15) < 0.1:
        phi_tabi[7] = 0.
    else:
        sum_CpdT = 0.
        dT = (Tin-298.15)/(int(100*abs(Tin-298.15))-1)
        for kk in range(1,int(100*abs(Tin-298.15))):
            gasCP.TPX = (298.15 + kk*dT),p,reactants
            cp_1 = gasCP.cp_mass
            sum_CpdT = sum_CpdT + 0.5*(cp_0 + cp_1)*dT
            cp_0 = cp_1
        phi_tabi[7] = sum_CpdT     



def multi_canSim(item):
    i, Z, phi, fuel_species, CASENAME, chemMech, transModel, nSpeMech, nVarCant, \
        p, Lx, stoich_O2, T_fuel, X_fuel, T_ox, X_ox, solFln = item
    Sim(i, Z, phi, fuel_species, CASENAME, chemMech, transModel, nSpeMech, nVarCant, \
        p, Lx, stoich_O2, T_fuel, X_fuel, T_ox, X_ox, solFln)

def canteraSim(cbDict,solFln):
    global phi_tab, sL

    # solFln = (os.getcwd() + '/canteraData/')
    # solFln = ('./canteraData/')
    if not os.path.isdir(solFln): os.mkdir(solFln)

    CASENAME = cbDict['CASENAME'] # case name
    p =  cbDict['p']  # pressure [Pa]
    Lx = cbDict['Lx'] # Domain size for the simulation [m]
    chemMech = cbDict['chemMech'] # chemical mechanism
    transModel = cbDict['transModel']
    nSpeMech=cbDict['nSpeMech']
    nVarCant=cbDict['nVarCant']

    ## Fuel characteristics
    fuel_species = cbDict['fuel_species'] # Fuel is assumed to be of the form CxHy
    fuel_C = cbDict['fuel_C'] # number of C atoms in the fuel
    fuel_H = cbDict['fuel_H'] # number of H atoms in the fuel
    stoich_O2 = fuel_C+fuel_H/4. # DO NOT CHANGE - stoich air mole fraction
    W_fuel = fuel_C * 12. + fuel_H * 1.0 # DO NOT CHANGE - fuel molar weight
    T_fuel = cbDict['T_fuel'] # Fuel temperature [K]
    X_fuel = cbDict['X_fuel'] # Fuel composition (in mole fraction)

    ## Oxidiser characteristics
    W_O2 = 2. * 16. # DO NOT CHANGE - molar weight of O2
    W_N2 = 2. * 14. # DO NOT CHANGE - molar weight of N2
    T_ox = cbDict['T_ox'] # oxidiser temperature [K]
    X_ox = cbDict['X_ox'] # oxidiser composition (in mole fraction)

    ## Mixture properties
    # DO NOT CHANGE - stoichiometric mixture fraction
    Zst = (W_fuel) / (W_fuel + stoich_O2 * ( W_O2 + 3.76 * W_N2) )
    # DO NOT CHANGE - array of mixture fraction of interest
    Z = np.linspace(cbDict['f_min'],cbDict['f_max'],cbDict['nchemfile'])  

    # DO NOT CHANGE BELOW THIS LINE
    phi = Z*(1.0 - Zst) / (Zst*(1.0 - Z))     #当量比phi
    # phi_tab = np.zeros((len(phi),8)) #各列为：当量比、混合物分数、网格点数、火焰速度、火焰厚度、热释放参数
    phi_tab=[]
    sL = []     #层流火焰速度
    for jj in range (cbDict['nchemfile']):
        phi_tab.append(Array('d',8,lock=False))
    

    nscal_BC = cbDict['nscal_BC']
    BCdata = np.zeros((2,nscal_BC+2*cbDict['nYis']))  

    fln_phi_tab = solFln + 'lamParameters.txt'    
    #在work_dir/canteraData/solution_00文件夹中创建文件lamParameters.txt    

    inputList = []
    for i in range(cbDict['nchemfile']):
        inputList.append( (i, Z, phi, fuel_species, CASENAME, \
                chemMech, transModel, nSpeMech, nVarCant, p, Lx, stoich_O2, \
                T_fuel, X_fuel, T_ox, X_ox, solFln ) )

    # with Pool(processes=cbDict['n_procs']) as pool:
    #     pool.map(Sim,range(cbDict['nchemfile']))
    with Pool() as pool:
        pool.map(multi_canSim,inputList)


    ###### calculate boundary conditions for pure fuel and oxidiser 
    gas_fuel = ct.Solution(chemMech,'gri30_multi')
    gas_fuel.TPX = T_fuel,p,X_fuel
    BCdata[0,0] = gas_fuel.T               
    #燃料温度
    BCdata[0,1] = gas_fuel.density_mass    
    #燃料密度

    gas_fuelCP = ct.Solution(chemMech)
    gas_fuelCP.TPX = 298.15,p,X_fuel
    cp_0 = gas_fuelCP.cp_mass
    if abs(T_fuel - 298.15) < 0.1:
        BCdata[0,2] = cp_0
    else:
        sum_CpdT = 0.                                       
        #integral of cp in T space from 298.15 to T_fuel
        dT = (T_fuel-298.15)/(int(100*abs(T_fuel-298.15))-1)
        for kk in range(1,int(100*abs(T_fuel-298.15))):
            gas_fuelCP.TPX = (298.15 + kk*dT),p,X_fuel
            cp_1 = gas_fuelCP.cp_mass
            sum_CpdT = sum_CpdT + 0.5*(cp_0 + cp_1)*dT
            cp_0 = cp_1
        BCdata[0,2] = sum_CpdT / (T_fuel-298.15)    
    #燃料的平均定压比热容（温度从298.15K到T_fuel）

    BCdata[0,3] = np.dot(np.transpose(gas_fuel.X),gas_fuel.molecular_weights) 
    #BCdata的第3列：燃料的质量   #燃料的摩尔分数转置后，与摩尔质量点积

    gas_fuelHf = ct.Solution(chemMech)
    gas_fuelHf.TPX = 298.15,p,X_fuel
    BCdata[0,4] = gas_fuelHf.enthalpy_mass         
    #燃料的焓

    BCdata[0,5] = gas_fuel.viscosity/gas_fuel.density_mass  
    #燃料的运动粘性系数
    BCdata[0,6] = gas_fuel.enthalpy_mass                    
    #燃料的焓  （'gri30_multi'）  #和第4列的区别是什么？

    print('T_fuel_approx: {0:7f}'.format((BCdata[0,6]-BCdata[0,4])
                                        /BCdata[0,2]+298.15))

    gas_ox = ct.Solution(chemMech,'gri30_multi')
    gas_ox.TPX = T_ox,p,X_ox
    BCdata[1,0] = gas_ox.T                 
    #氧化剂温度
    BCdata[1,1] = gas_ox.density_mass      
    #氧化剂密度

    gas_oxCP = ct.Solution(chemMech)
    gas_oxCP.TPX = 298.15,p,X_ox
    cp_0 = gas_oxCP.cp_mass
    if abs(T_ox - 298.15) < 0.1:
        BCdata[1,2] = cp_0
    else:
        sum_CpdT = 0.0
        dT = (T_ox-298.15)/(int(100*abs(T_ox-298.15))-1)
        for kk in range(1,int(100*abs(T_ox-298.15))):
            gas_oxCP.TPX = (298.15 + kk*dT),p,X_ox
            cp_1 = gas_oxCP.cp_mass
            sum_CpdT = sum_CpdT + 0.5*(cp_0 + cp_1)*dT
            cp_0 = cp_1
        BCdata[1,2] = sum_CpdT / (T_ox-298.15)     
    #氧化剂的平均定压比热容

    BCdata[1,3] = np.dot(np.transpose(gas_ox.X),gas_ox.molecular_weights)   
    #氧化剂的质量

    gas_oxHf = ct.Solution(chemMech)
    gas_oxHf.TPX = 298.15,p,X_ox
    BCdata[1,4] = gas_oxHf.enthalpy_mass     
    #氧化剂的焓

    BCdata[1,5] = gas_ox.viscosity/gas_ox.density_mass    
    #氧化剂的运动粘性系数
    BCdata[1,6] = gas_ox.enthalpy_mass                    
    #氧化剂的焓  （'gri30_multi'）

    print('T_ox_approx: {0:7f}'.format((BCdata[1,6]-BCdata[1,4])/BCdata[1,2]
                                        +298.15))

    # species BCs
    gas = ct.Solution(chemMech)
    for s in range(cbDict['nYis']):
        ispc = gas.species_index(cbDict['spc_names'][s])    
        #第s个组分的索引
        iBC = (nscal_BC-1) + s*2 + 1  
        #本循环中的结果存储在BCdata的最后2*nYis列  
        BCdata[0,iBC] = ispc     
        #第s个组分的索引
        BCdata[1,iBC] = ispc     
        #第s个组分的索引
        iBC = (nscal_BC-1) + (s+1)*2
        if gas_fuel.Y[ispc]>1.e-30:
            BCdata[0,iBC] = gas_fuel.Y[ispc]    
            #燃料中第s个组分的质量分数
        if gas_ox.Y[ispc]>1.e-30:
            BCdata[1,iBC] = gas_ox.Y[ispc]      
        #氧化剂中第s个组分的质量分数


    # save the laminar parameters of all the flamelets
    fmt_str1=''
    for ff in range(8):  
        if ff == 2:
            fmt_str1 = fmt_str1 + '%04d '
        else:
            fmt_str1 = fmt_str1 + '%.5e '
        #
    fmt_str2=''
    for ff in range(len(BCdata[0,:])):    
        nn = ff - nscal_BC
        if nn >= 0 and nn % 2 == 0:
            fmt_str2 = fmt_str2 + '%04d '
        else:
            fmt_str2 = fmt_str2 + '%.5e '
        #

    with open(fln_phi_tab,'w') as strfile:
        strfile.write(CASENAME + '\n')
        np.savetxt(strfile,phi_tab,fmt=fmt_str1.strip())
        np.savetxt(strfile,BCdata,fmt=fmt_str2.strip())
    strfile.close()                               
    #将phi_tab、BCdata保存到lamParameters.txt
    #

    # # 将canteraData/文件夹压缩打包
    # with zipfile.ZipFile('canteraData.zip','w') as target:
    #     for i in os.walk(solFln):
    #         for n in i[2]:
    #             target.write(''.join((i[0],n)))

    # zipPath='canteraData.zip'

    # return zipPath

    # tarPath = 'canteraData.tar'
    # with tarfile.open(tarPath, 'w') as tar:
    #     tar.add(solFln)

    # return tarPath

    # return solFln


