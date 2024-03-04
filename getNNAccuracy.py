import numpy as np
from prepareTarget import prepareTarget

def getNNAccuracy(net, data):
    x = data[:, :-1].T  # Transpose X to match MATLAB's behavior
    t = prepareTarget(data[:, -1]).T
    
    y = net(x)
    tind = np.argmax(t, axis=0) + 1  # Add 1 to match MATLAB's indexing
    yind = np.argmax(y, axis=0) + 1  # Add 1 to match MATLAB's indexing
    acc = np.mean(yind == tind)
    return acc
