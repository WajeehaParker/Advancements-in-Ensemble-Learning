import numpy as np

def prepareTarget(Y):
    dim = np.unique(Y)
    id_matrix = np.eye(len(dim))
    target = np.zeros((len(Y), len(dim)))
    
    for i in range(len(Y)):
        for j in range(len(dim)):
            if Y[i] == dim[j]:
                target[i, :] = id_matrix[j, :]
                break  # Break out of the loop once target is set
    
    return target
