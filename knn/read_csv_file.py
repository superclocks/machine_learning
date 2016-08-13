import os
import csv
import shutil

import numpy as np
from sklearn.datasets import base 

    
def loadSamples(path, samples, features, read_target = False, target_place = 'first'):
    
    module_path = path
    data_file = csv.reader(open(path))
    #fdescr = open(join(module_path, 'descr', 'iris.rst'))
    temp = next(data_file)
    n_samples = samples
    n_features = features
    
    data = np.empty((n_samples, n_features))
    
    if (read_target):
        target = np.empty((n_samples,), dtype=np.int)
    
        for i, ir in enumerate(data_file):
            if target_place == 'first':
                data[i] = np.asarray(ir[1:], dtype=np.float)
                target[i] = np.asarray(ir[0], dtype=np.int)                
            else:
                data[i] = np.asarray(ir[:-1], dtype=np.float)
                target[i] = np.asarray(ir[-1], dtype=np.int)
    
        return base.Bunch(data=data, target=target)
    else:            
        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float)
    
        return base.Bunch(data=data)        

if __name__ == "__main__":
    r = loadSamples('./data.csv',150,4,True,'first')
    X = r.data
    Y = r.target
    
    rr = loadSamples('./data.csv',150,4)
    test = rr.data