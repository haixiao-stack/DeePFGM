import imp
from numpy import array, double, float64
from pathlib import Path
import time
from typing import List
import ast
import os
import numpy as np
if __name__ == "__main__":

    #-------------load commonDict.txt--------------#
    
    start=time.time()

    
    import read_commonDict
    import interpToMeshgrid
    import multiProcessing_canteraSim
    import assemble
    import PDF_Sim
    cbDict = read_commonDict.read_commonDict("commonDict.txt")
    # print(cbDict['c'])
    # print(cbDict['work_dir'])
    solFln = ('./canteraData/')
    multiProcessing_canteraSim.canteraSim(cbDict,solFln)
    interpToMeshgrid.interpLamFlame(cbDict,solFln)
    # os.chdir('./canteraData/')
    cbDict['z'] = np.append([0],cbDict['z'])
    cbDict['c'] = np.append([0],cbDict['c'])
    cbDict['gz'] = np.append([0],cbDict['gz'])
    cbDict['gc'] = np.append([0],cbDict['gc'])
    cbDict['gcz'] = np.append([0],cbDict['gcz'])
    # PDF_Sim.multiprocessingpdf(cbDict)
    assemble.assemble(cbDict,solFln)
    end=time.time()
    print('Running time: %s Seconds'%(end-start))


