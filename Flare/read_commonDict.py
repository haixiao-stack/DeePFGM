def read_commonDict(commonDict_path):
    import numpy as np
    import cantera as ct
    import ast

    #---------read commonDict.txt--------#
    with open(commonDict_path,"r") as f:
        commonDict = f.read()
        f.close()

    cbDict = ast.literal_eval(commonDict)

    #Total number of species in all phases participating in the kinetics mechanism
    cbDict['nSpeMech'] = ct.Solution(cbDict['chemMech']).n_total_species

    #----------create meshgrid------------#
    z_space = np.linspace(0,1,cbDict['n_points_z']) 
    nn = int(cbDict['n_points_z']/20*19)
    for i in range(nn):
        z_space[i] = cbDict['f_max']*1.2/(nn-1)*i
    z_space[nn-1:] = np.linspace(cbDict['f_max']*1.2,1.0,
                                cbDict['n_points_z']-nn+1) 

    c_space = np.linspace(0,1,cbDict['n_points_c'])

    cbDict['z_space'] = z_space
    cbDict['c_space'] = c_space

    #----------create manifold------------#
    z_int = np.linspace(0,1,cbDict['int_pts_z']) 
    nn = int(cbDict['int_pts_z']/20*19)            
    for i in range(nn):
        z_int[i] = cbDict['f_max']*1.2/(nn-1)*i
    z_int[nn-1:] = np.linspace(cbDict['f_max']*1.2,1.0,
                                cbDict['int_pts_z']-nn+1)

    c_int = np.linspace(0,1,cbDict['int_pts_c'])  

    gz_int = np.zeros(cbDict['int_pts_gz'])
    gz_int[1:] = np.logspace(-4,-1,cbDict['int_pts_gz']-1)  
    # gz_int[1:] = np.logspace(-4,0,cbDict['int_pts_gz']-1)  

    gc_int = np.linspace(0,1-cbDict['small'],cbDict['int_pts_gc'])   
    # gc_int = np.linspace(0,1,cbDict['int_pts_gc'])   

    # gcor_int = np.linspace(-1,1,cbDict['int_pts_gcor']) 
    if (cbDict['int_pts_gcz'] <= 1):
        gcz_int = np.linspace(0,0,cbDict['int_pts_gcz'])
    else:
        gcz_int = np.linspace(-1,1,cbDict['int_pts_gcz'])   

    #save to cbDict
    cbDict['z'] = z_int
    cbDict['c'] = c_int
    cbDict['gz'] = gz_int
    cbDict['gc'] = gc_int
    # cbDict['gcor'] = gcor_int
    cbDict['gcz'] = gcz_int

    return cbDict