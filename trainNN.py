from sklearn.neural_network import MLPClassifier
from prepareTarget import prepareTarget

def trainNN(X, y):
    print("In trainNN")
    x = X.T  # Transpose X
    t = prepareTarget(y).T  # Transpose prepared target
    net = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    net.fit(x, t)
    return net
