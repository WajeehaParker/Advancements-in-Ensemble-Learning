import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def dataNormalize(D, type):
    if type == 1:
        max_x = np.max(D, axis=0)
        min_x = np.min(D, axis=0)
        
        if np.any(max_x - min_x == 0):
            return D
        
        X = (D - min_x) / (max_x - min_x)
        
    elif type == 2:
        mean_x = np.mean(D, axis=0)
        std_x = np.std(D, axis=0)
        
        if np.any(mean_x == 0):
            return D
        
        X = (D - mean_x) / std_x
        
    elif type == 3: # SAMS NORM
        scaler = MinMaxScaler()
        X = scaler.fit_transform(D)
        
    else:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(D)
        
    return X
