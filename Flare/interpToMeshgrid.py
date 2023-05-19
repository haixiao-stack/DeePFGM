import numpy as np
import numpy.matlib
import scipy
import tarfile
import os
# import zipfile

def interpLamFlame(cbDict,solFln):

    # tar = tarfile.open(solFln,'r')
    # tar.extractall()
    # tar.close()

    # solFln = "./canteraData"

    # os.chdir(solFln)

    lamArr = []                                     
    #存放从lamParameters.txt中读取的phi_tab的值 
    with open(solFln + '/' + 'lamParameters.txt') as f:   
        ll = 0
        for line in f:
            if ll == 0:
                casename = line.strip()     
                #Python strip() 方法用于移除字符串头尾指定的字符
                # （默认为空格或换行符）或字符序列。
            elif ll > cbDict['nchemfile']:  
              #只读lamParameters.txt中前nchemfile行数据    
              #nchemfile：需要计算的当量比的数量
                break
            else:
                line = line.strip()
                lamArr.append(line.split(' '))   
                #Python split() 通过指定分隔符对字符串进行切片，
                #如果参数 num 有指定值，则分隔 num+1 个子字符串
            ll += 1
    lamArr = np.array(lamArr,dtype='float')    
    #将lamArr的数据创建为一个数组lamArr，数据类型为float

    streamBC = np.loadtxt(solFln + '/' + 'lamParameters.txt',
                          skiprows=1+cbDict['nchemfile'])    
                          #将边界条件数据BCdata读取到streamBC中

    with open(solFln + '/' + cbDict['output_fln'] ,'w') as strfile:             
        strfile.write('%.5E' % streamBC[0,6] + '\t' +
                    '%.5E' % streamBC[1,6] + '\n')    
                    #燃料/氧化剂的焓  （'gri30_multi'）
        strfile.write(str(cbDict['nchemfile']) + '\n')
        np.savetxt(strfile,
                    np.transpose([lamArr[:,1],lamArr[:,3],lamArr[:,4],
                                lamArr[:,5],lamArr[:,6]]),
                    fmt='%.5E',delimiter='\t')         
                    #当量比、网格点数、火焰速度、火焰厚度、热释放参数
    strfile.close()

    # read cantera solutions & calculate c,omega_c
    nScalCant = cbDict['nSpeMech'] + cbDict['nVarCant']    
    #总组分数+nVarCant  #nVarCant=10
    mainData = np.zeros((cbDict['nchemfile'],int(max(lamArr[:,2])),nScalCant))  
    #矩阵大小：nchemfile*max(lamArr[:,2])*nScalCant   
    #nchemfile：要计算的当量比数目  #lamArr[:,2]：混合物分数
    cIn = np.zeros(np.shape(mainData[:,:,0]))    
    #np.shape(mainData[:,:,0])：maindata当nScalCant=0时的维数，即：nchemfile*max(lamArr[:,2])
    omega_cIn = np.zeros(np.shape(mainData[:,:,0]))    
    #矩阵大小：nchemfile*max(lamArr[:,2])
    Yc_eq = np.zeros((cbDict['nchemfile'])) 
    for i in range(cbDict['nchemfile']):    #对于每个当量比phi：
        fln = (solFln + '/' + casename + '_' + str('{:03d}'.format(i)) + '.csv')
        print('\nReading --> ' + fln)  #读取对应的文件CH4_i.csv到fln中

        len_grid = int(lamArr[i,2])   #网格点数
        with open(fln) as f:
            j = 0
            for line in f:
                if j >= len_grid:
                    break
                line = line.strip()
                mainData[i,j,:] = line.split(' ') #将CH4_i.csv的第j行值赋给mainData[i,j,:]
                j += 1

            # for j in range(int(max(lamArr[:,2]))):
            #         if j < int(lamArr[i,2]):
            #             mainData[i,j,:] = np.loadtxt(fln,skiprows=j,max_rows=1)
            #             end = j
            # else: mainData[i,j,:] = np.loadtxt(fln,skiprows=end,max_rows=1)

        imax = np.argmax(mainData[i,:,0])      
        #mainData[i,:,0]最大值的索引   #mainData[i,:,0]:CO和CO2的质量分数之和表示的反应进度
        cIn[i,:] = mainData[i,:,0] / mainData[i,imax,0]     
        #反应进度的向量/最大反应进度
        if mainData[i,imax,0]/mainData[i,len_grid-1,0] > 1.0:    
          #反应过度
            print('c_max/c_end =',mainData[i,imax,0]/mainData[i,len_grid-1,0],
                  ' --> overshooting')

        if(cbDict['scaled_PV']):             
          #是否使用相对的反应进度   
          Yc_eq[i] = mainData[i,imax,0]
          omega_cIn[i,:] = mainData[i,:,1] / Yc_eq[i]    
          #反应速率向量/最大反应进度
        else:
          omega_cIn[i,:] = mainData[i,:,1]

    d2Yeq_table = []
    if(cbDict['scaled_PV']):
        d2Yeq_table = generateTable2(lamArr[:,1],Yc_eq,cbDict["z"],cbDict["gz"],cbDict["f_min"],cbDict["f_max"])     
      #如果使用相对值，调用generateTable2函数，计算截断后的进程变量Z1的概率密度分布    
      #输入：lamArr[:,1] 混合物分数； Yc_eq 反应速率； contVarDict 初始的z、c、gz、gc、gcz(都是向量)
        fln = (solFln + '/' + 'd2Yeq_table.dat')  
        #work_dir/canteraData/solution_00文件夹下创建d2Yeq_table.dat文件
        print('\nWriting --> ' + fln)
        np.savetxt(fln,d2Yeq_table,fmt='%.5E')   

    # interpolate in c space & write out for each flamelet
    MatScl_c = np.zeros((cbDict['nchemfile'],cbDict['cUnifPts'],      
                         nScalCant+1))
            #chemfile*cUnifPts*(nScalCant+1)    #chemfile:当量比的数量  #cUnifPts：用进程变量c插值的节点数  
            #nScalCant = nSpeMech+nVarCant
    for i in range(cbDict['nchemfile']):   
      #对于每个当量比i：
        len_grid = int(lamArr[i,2])        
        #网格点数
        ctrim = cIn[i,:len_grid-1]         
        #反应进度cIn的第i行0:(len_grid-2)列
        # 0:c|1:omg_c
        MatScl_c[i,:,0] = np.linspace(0.,1.,cbDict['cUnifPts'])      
        #进程变量c的插值点     #当量比i下的MatScl_c的第0列：0,...,1   cUnifPts个点
        MatScl_c[i,:,1] = np.matlib.interp(MatScl_c[i,:,0],ctrim,
                                         omega_cIn[i,:len_grid-1])   
                                #当量比i下的MatScl_c的第1列：MatScl_c[i,:,0]插值得到的反应速率    
                                #MatScl_c[i,:,0]在x=ctrim, y=omega_cIn[i,:len_grid-1]上插值
        # 2:T|3:rho|4:cp|5:mw|6:hf_0|7:nu|8:h|9:qdot
        for k in range(2,nScalCant):
            MatScl_c[i,:,k] = np.matlib.interp(MatScl_c[i,:,0],ctrim,
                                             mainData[i,:len_grid-1,k])   
                #当量比i下的MatScl_c的第2到nScalCant列：插值得到的温度/密度/定压比热容/质量/生成焓/运动粘度/焓/释热率
        # cp-->cp_e
        MatScl_c[i,:,4] = calculateCp_eff(MatScl_c[i,:,2],MatScl_c[i,:,4],
                                          lamArr[i,7])    
            #把定压比热容改成有效定压比热容  #输入：MatScl_c[i,:,2]温度  MatScl_c[i,:,4]定压比热容  
            #lamArr[i,7]定压比热容在T=298.15~Tin的积分
        # Yc_max
        MatScl_c[i,:,-1] = mainData[i,len_grid-1,0]    
        #当量比i下的MatScl_c的最后一列：出口处的反应进度

        # write inpterpolated 1D profiles
        fln = (solFln + '/' + 'Unf_'+casename+'_'+str('{:03d}'.format(i))+'.dat')  
        #work_dir/canteraData/solution_00文件夹下创建Unf_CH4_i.dat文件
        print('\nWriting --> ' + fln)
        np.savetxt(fln,MatScl_c[i,:,:],fmt='%.5e')         
        #将MatScl_c第i个当量比的数据写入Unf_CH4_i.dat

    oriSclMat = np.zeros([cbDict['nchemfile']+2,cbDict['cUnifPts'],
                          nScalCant+1]) 

    ind_list_Yis = []
    for i in range(len(streamBC[:,0])):   
      #len(streamBC[:,0])=2

        if i == 0: j = len(oriSclMat[:,0,0]) - 1   
        #若i==0,则j=nchemfile+1
        else: j = 0                                
        #else, j=0
        for k in range(len(streamBC[0,:])):        
          #len(streamBC[0,:])=nscal_BC+2*nYis
            # for thermo scalars
            if k < len(streamBC[0,:]) - 2*cbDict['nYis']:  
              #若k<nscal_BC:
                oriSclMat[j,:,k+2] = streamBC[i,k]         
                #i=0,j=nchemfile+1=51时,oriSclMat[j,:,k+2]存放:
                #燃料的温度、密度、平均定压比热容、质量、焓、运动粘度、焓  （'gri30_multi'）
                #i=1,j=0时，oriSclMat[j,:,k+2]存放氧化剂的相应值
            else:
                nk = k                       
                #当循环到k=nscal_BC时, 令nk=nscal_BC,并终止k的循环
                break

        # for selected species
        for s in range(cbDict['nYis']):     
          #'H2O','CO','CO2'   nYis=3
            ispc=int(streamBC[i,nk+s*2])     
            #ispc=int(BCdata[i,nscal_BC+s*2])     #燃料/氧化剂中第s个组分的索引
            if i == 0: ind_list_Yis.append(ispc)    
            #若i=0, 则把ispc添加入ind_list_Yis
            iscal = cbDict['nVarCant']+ispc         
            #nVarCant=10
            oriSclMat[j,:,iscal]=streamBC[i,nk+s*2+1]   
            #i=0,j=nchemfile+1=51时，oriSclMat[j,:,iscal]存储第s个燃料的质量分数；i=1,j=0时，存放第s个氧化剂的质量分数

    oriSclMat[1:cbDict['nchemfile']+1,:,:] = MatScl_c    
    #oriSclMat的中间部分存放MatScl_c的结果

    intpSclMat = np.zeros([len(cbDict['z_space']),len(cbDict['c_space']),
                           cbDict['nScalars']])    
                           #大小：z_space*c_space*nScalars
    intpYiMat = np.zeros([len(cbDict['z_space']),len(cbDict['c_space']),
                        cbDict['nYis']])           
                        #大小：z_space*c_space*nYis
    Z_pts = np.insert(lamArr[:,1],0,[0.],axis=0)   
    #在混合物分数向量的最前面插入值0.0     #lamArr[:,1] 混合物分数
          #numpy.insert(arr,obj,values,*axis)在arr的obj位置插入values  arr:1维或多维数组，在arr基础上插入元素；obj：元素插入的位置；values：需要插入的数值；axis：可选参数，在哪个轴上对应的插入位置进行插入
    Z_pts = np.insert(Z_pts,len(Z_pts),[1.],axis=0)  
    #再在混合物分数向量的最后面插入值1.0
    c_pts = MatScl_c[0,:,0]    
    #最小当量比时的进程变量

    np.array(ind_list_Yis)    
    #将ind_list_Yis转为array类型

    print('\nInterpolating...')
    intpSclMat[:,:,0] = interp2D(oriSclMat[:,:,3],Z_pts,c_pts,cbDict['z_space'],cbDict['c_space']) # rho        
    #对在（Z_pts,c_pts）分布的密度oriSclMat[:,:,3]进行插值，插值点为meshgrid （z_space,c_space）
    for k in [1,4,5,6]: # omega_c,cp_e,mw,hf                                  
        intpSclMat[:,:,k] = interp2D(oriSclMat[:,:,k],Z_pts,c_pts,cbDict['z_space'],cbDict['c_space'])          
        #对反应速率、cp_e,质量，生成焓插值
    intpSclMat[:,:,7] = interp2D(oriSclMat[:,:,2],Z_pts,c_pts,cbDict['z_space'],cbDict['c_space']) # T          
    #对温度插值
    intpSclMat[:,:,8] = interp2D(oriSclMat[:,:,7],Z_pts,c_pts,cbDict['z_space'],cbDict['c_space']) # nu         
    #对运动粘度插值
    intpSclMat[:,:,9] = interp2D(oriSclMat[:,:,8],Z_pts,c_pts,cbDict['z_space'],cbDict['c_space']) # h          
    #对焓插值
    intpSclMat[:,:,10] = interp2D(oriSclMat[:,:,9],Z_pts,c_pts,cbDict['z_space'],cbDict['c_space']) # qdot
    #对释热率插值

    # Yc_max
    if cbDict['scaled_PV']:           
      #如果使用相对反应进度
      intpSclMat[:,:,11] = 1.0        
      #intpSclMat[:,:,11]设为1.0
    else:
      intpSclMat[:,:,11] = interp2D(oriSclMat[:,:,-1],Z_pts,c_pts,
                                    cbDict['z_space'],cbDict['c_space'])   
                                    #否则，对出口处的反应进度插值    #表示进程变量的最大值  

    for y in range(cbDict['nYis']):
        iy = ind_list_Yis[y]    
        #燃料中的第y个组分的索引
        intpYiMat[:,:,y] = interp2D(oriSclMat[:,:,iy+cbDict['nVarCant']],
                                            Z_pts,c_pts,cbDict['z_space'],cbDict['c_space'])    
                                            #对燃料/氧化剂的质量分数插值
    print('\nInterpolation done. ')

    if np.sum(np.isnan(intpSclMat)) > 0:   
      #np.isnan(intpSclMat)：检查intpSclMat是否为数字，如果是，则返回True，否则返回False
        print('\nNumber of Nans detected: ', np.sum(np.isnan(intpSclMat)))    
        #np.sum(): 求所有元素之和，此处用于计数
    else: print('\nNo Nans detected. Well done!')

    print('\nwriting chemTab file...')
    Arr_c,Arr_Z = np.meshgrid(cbDict['c_space'],cbDict['z_space'])   
    #插值点：混合物分数c、进程变量z
    Idx_outLmt = np.hstack([(Arr_Z>cbDict['f_max']).nonzero(),
                             (Arr_Z<cbDict['f_min']).nonzero()])   
            #在Arr_Z=(f_min,f_max)范围之外的数据的索引     #np.hstack()：按水平顺序连接数组  
            #np.nonzero()：用于得到数组array中粉岭元素的位置（数组索引）
    ind_rates=[1,9]                                                
    intpSclMat[:,:,ind_rates][Idx_outLmt[0],Idx_outLmt[1]] = 0.    
    #将超出Arr_Z=(f_min,f_max)范围的部分在intpSclMat的反应速率、焓设为0
    chemMat = np.append(intpSclMat,intpYiMat,axis=2)     
    #在chemMat中添加intpSclMat,intpYiMat的结果       #为什么令axis=2？
    chemMat = np.insert(chemMat,0,Arr_c,axis=2)          
    #在chemMat的最前面添加混合物分数Arr_c
    chemMat = np.insert(chemMat,0,Arr_Z,axis=2)          
    #在chemMat的最前面再添加进程变量Arr_Z    #chemMat的大小：z_space*c_space*（nScalars+nYis+2）
    stackMat = np.reshape(chemMat,[np.shape(chemMat)[0]*np.shape(chemMat)[1],
                              np.shape(chemMat)[2]])     
            #将chemMat转换成大小为(np.shape(chemMat)[0]*np.shape(chemMat)[1]，
            #np.shape(chemMat)[2])=（z_space*c_space，nScalars+nYis+2）的形式   

    fln = solFln + '/' + 'chemTab_' + str('{:02d}'.format(cbDict['solIdx'])) + '.dat'   
    #创建文件work_dir/chemTab_01.dat   #solIdx=1
    np.savetxt(fln,stackMat,fmt='%.5E')    
    #将stackMat保存到chemTab_01.dat


    print('\ninterpToMeshgrid Done.')

    # with zipfile.ZipFile('interpToMeshgrid.zip','w') as target:
    #     for i in os.walk('./'):
    #         for n in i[2]:
    #             target.write(''.join((i[0],n)))

    # zipPath='canteraData.zip'

    # return zipPath

    #经过比较，用tar打包速度更快

    # tarPath = 'interpData.tar'
    # with tarfile.open(tarPath, 'w') as tar:
    #     tar.add('./' + cbDict['output_fln'])
    #     tar.add('./d2Yeq_table.dat')
    #     tar.add('./chemTab_' + str('{:02d}'.format(cbDict['solIdx'])) + '.dat')
    #     for i in range(cbDict['nchemfile']):   
    #         tar.add('./Unf_'+casename+'_'+str('{:03d}'.format(i))+'.dat')

        # tar.add('.')

    # return tarPath

    # return solFln

''' ===========================================================================

Subroutine functions

=========================================================================== '''

def calculateCp_eff(T,cp_m,lam_sumCpdT):     
    #输入：MatScl_c[i,:,2]温度  MatScl_c[i,:,4]定压比热容  lamArr[i,7]定压比热容在T=298.15~Tin的积分  i当量比
    cp_e = np.zeros(np.shape(cp_m))   
    T_0 = 298.15 
    if abs(T[0] - T_0) > 0.1: cp_e[0] = lam_sumCpdT / (T[0] - T_0)   
    #如果初始温度比298.15大0.1，则cp_e第0项为平均比热容
    else: cp_e[0] = cp_m[0]     #否则，为初始的MatScl_c[i,0,4]（定压比热容）
    for ii in range(1,len(T)):     
        #对于每个温度ii：
        if abs(T[0] - T_0) < 0.1: 
            tmp_sum=0.0
        else:
            tmp_sum = 0.
            for j in range(1,ii+1):
                tmp_sum = (tmp_sum + 0.5*(cp_m[j]+cp_m[j-1])*(T[j]-T[j-1]))  
            #从298.15到T吸收的总热量：tem_sum=Sum( 0.5*(cp_m[j]+cp_m[j-1]) *(T[j]-T[j-1]) )
        cp_e[ii] = (tmp_sum + lam_sumCpdT) / (T[ii] - T_0)
        #有效热容        
    return cp_e

def interp2D(M_Zc,Z_pts,c_pts,z_space,c_space):   
    #对于(Z_pts,c_pts)上取值的M_Zc，在插值点meshgrid上插值
    f = scipy.interpolate.interp2d(c_pts,Z_pts,M_Zc,kind="linear")
    intpM_Zc = f(c_space,z_space)
    # import matplotlib.pyplot as plt
    # plt.plot(c_pts, M_Zc[0, :], 'ro-', c_space, intpM_Zc[0, :], 'b-')
    # plt.show()
    return intpM_Zc

def generateTable2(lamArr,Yc_eq,z,gz,f_min,f_max):     
    #输入：lamArr[:,1] 混合物分数； Yc_eq 反应速率； contVarDict 初始的z、c、gz、gc、gcz
    Z0 = lamArr          
    #混合物分数
    Yc_eq0 = Yc_eq       
    #反应速率

    from scipy.interpolate import UnivariateSpline
    sp = UnivariateSpline(Z0,Yc_eq0,s=0)    
    #样条曲线y=sp(x)拟合到（x,y）=(Z0,Yc_eq0)上   返回值相当于一个函数
    Z_low_cutoff = f_min            
    #截断时，混合物分数Z0的最小阈值, Z_low_cutoff>=0
    Z_high_cutoff = f_max          
    #截断时，混合物分数Z0的最大阈值，Z_high_cutoff<=1
    Z1 = np.linspace(Z0[0],Z_high_cutoff,101)     
    #Z1=Z0[0],...,Z_high_cutoff   #101个节点
    Yc_eq1 = sp(Z1)                               
    #用拟合出的样条函数sp得到插值的反应速率

    import matplotlib.pyplot as plt
    plt.plot(Z0,Yc_eq0,label='original')
    plt.plot(Z1,Yc_eq1,label='spline')
    plt.legend()
    plt.show()

    d2 = sp.derivative(n=2)   
    #sp对x求二阶导
    d2Yc_eq1 = d2(Z1)         
    #混合物分数的二阶导
    sp = UnivariateSpline(Z1,d2Yc_eq1)     
    #混合物分数的二阶导的拟合曲线
    d2Yc_eq2 = sp(Z1)         
    #混合物分数的二阶导的拟合结果

    from scipy.signal import savgol_filter     
    #Savitzky-Golay滤波器，用于数据流平滑除噪，在时域内基于局部多项式最小二乘法拟合的滤波方法。
    # 特点：在滤除噪声的同时保持信号的形状、宽度不变
    #scipy.signal.savgol_filter(x,window_length,polyorder)；x为要滤波的信号；
    #window_length为窗口长度，取值为奇数且不能超过len(x)，越大则平滑效果越明显；
    #polyorder为多项式拟合的阶数，越小则平滑效果越明显
    d2Yc_eq3 = savgol_filter(d2Yc_eq2, 11, 3)   
    #对d2Yc_eq2滤波，window_length=11，polyorder=3
    plt.plot(Z1,d2Yc_eq1,label='original')      
    plt.plot(Z1,d2Yc_eq3,label='spline')
    plt.legend()
    plt.show()

    z_int = z           
    #导入z的初始值
    gz_int = gz         
    #导入gz的初始值
    gradd2 = np.gradient(d2Yc_eq3,Z1[1]-Z1[0])   
    #求d2Yc_eq3的梯度，d2Yc_eq3相邻元素之间的间距为Z1[1]-Z1[0]   
    #numpy.gradient(f,*varages)  f:一个包含标量函数样本的N-dimensional数组；varages：可选参数，f值之间的间距

    from scipy.stats import beta
    d2Yeq_int = np.zeros((len(z_int),len(gz_int)))  
    #矩阵大小len(z_int)*len(gz_int)=int_pts_z*int_pts_gz
    for i in range(1,len(z_int)-1):   
        #d2Yeq_int[0,:]本来就是0
        if z_int[i] > Z_high_cutoff or z_int[i] < Z_low_cutoff:
            d2Yeq_int[i,:] = 0.0           
            #在z的截断区间（Z_low_cutoff，Z_high_cutoff）之外的，设为0
        else:
            d2Yeq_int[i,0] =np.interp(z_int[i],Z1,d2Yc_eq3)   
            #点x=z_int[i]插值曲线d2Yc_eq3=f(Z1)上插值    #d2Yc_eq3：混合物分数的二阶导的拟合后再滤波的结果
            if (gz_int[-1]==1): gz_len = len(gz_int)-1
            else: gz_len = len(gz_int)
            for j in range(1,gz_len):      
            #表格d2Yeq_int的y方向
                a = z_int[i]*(1.0/gz_int[j]-1.0)  
                    #z*(1/gz-1)
                b = (1.0 - z_int[i])*(1.0/gz_int[j]-1.0)  
                #(1-z)*(1/gz-1)
                Cb = beta.cdf(Z1,a,b)    
                #累积分布函数（F_X(x)=P(X<=x),表示：对离散变量而言，所有小于等于X的值出现概率之和）  
                #a和b是形状参数，beta.cdf中计算了gamma(a),gamma(b),gamma(b)  #0<=Z1<=1
                d2Yeq_int[i,j] = d2Yc_eq3[-1] - np.trapz(gradd2*Cb,Z1)   
                #物理意义是：进程变量Z1的概率密度分布    #d2Yc_eq3[-1]：出口处混合物分数的二阶导  
                #gradd2：混合物分数的二阶导的梯度  #Cb：Z1的累积分布函数  #Z1：要拟合的混合物分数节点向量  
                #索引[-1]指向的是向量的倒数第一个值

    d2Yeq_int_1D = np.zeros((d2Yeq_int.flatten()).shape)   
    #d2Yeq_int.flatten()：把d2Yeq_int降到一维，默认按照行的方向降(第一行-第二行-...)  #.shape: 读取矩阵在各维度的长度
    count = 0
    for i in range(len(z_int)):
        for j in range(len(gz_int)):
            d2Yeq_int_1D[count] = d2Yeq_int[i][j]   
            #把进程变量Z1的概率密度分布矩阵d2Yeq_int排成向量赋值给d2Yeq_int_1D，返回d2Yeq_int_1D的值
            count = count + 1

    return d2Yeq_int_1D

