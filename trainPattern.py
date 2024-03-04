from sklearn.neural_network import MLPClassifier
from prepareTarget import prepareTarget

def trainPattern(X, y):
    y = prepareTarget(y).T  # Transpose and prepare target
    net = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)
    net.fit(X, y)
    return net
