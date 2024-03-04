# from sklearn.neural_network import RBFClassifier
# from prepareTarget import prepareTarget

# def trainRBFNN(X, y, params, p):
#     x = X
#     t = prepareTarget(y).T  # Transpose and prepare target
#     eg = 0.03  # sum-squared error goal
#     sc = 1     # spread constant
#     net = RBFClassifier(eta0=0.1, max_iter=1000)  # Example parameters; adjust as needed
#     net.fit(x, t)
#     return net
