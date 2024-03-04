import numpy as np

def getNNPredict(net, X):
    x = X.T  # Transpose X to match MATLAB's behavior
    y = net(x)
    predict = np.argmax(y, axis=0) + 1  # Add 1 to match MATLAB's indexing
    return predict
