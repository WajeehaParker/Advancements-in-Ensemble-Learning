import numpy as np

def getBalance(Ytrain):
    y = Ytrain
    classes = np.unique(y)
    noOfClasses = len(np.unique(y))
    a = np.zeros((1, noOfClasses))
    index = 0
    
    for i in range(len(classes)):
        a[0, index] = np.sum(y == classes[i]) / len(y)
        index += 1
    
    stdev = np.std(a)
    
    d = a.shape[1]
    l2 = (np.linalg.norm(a) * np.sqrt(d) - 1) / (np.sqrt(d) - 1)
    
    return stdev, l2
