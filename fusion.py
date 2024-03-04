import numpy as np
from getNNPredict import getNNPredict
from scipy.stats import mode

def fusion(classifiers, data):
    X = data[:, :-1]
    Y = data[:, -1]
    decisionMatrix = np.ones((len(X), len(classifiers)))
    index = 0
    
    for i in range(len(classifiers)):
        try:
            if classifiers[i]['name'] == 'SVM':
                decisionMatrix[:, index] = classifiers[i]['model'].predict(X)
            elif classifiers[i]['name'] == 'KNN':
                decisionMatrix[:, index] = classifiers[i]['model'].predict(X)
            elif classifiers[i]['name'] == 'DT':
                decisionMatrix[:, index] = classifiers[i]['model'].predict(X)
            elif classifiers[i]['name'] == 'NB':
                decisionMatrix[:, index] = classifiers[i]['model'].predict(X)
            elif classifiers[i]['name'] == 'DISCR':
                decisionMatrix[:, index] = classifiers[i]['model'].predict(X)
            elif classifiers[i]['name'] == 'ANN':
                decisionMatrix[:, index] = getNNPredict(classifiers[i]['model'], X)
            index += 1
        except Exception as ME:
            print(f'Fusion causing errors: {ME}')
    
    decisionMatrix = mode(decisionMatrix, axis=1)[0]
    acc = np.mean(decisionMatrix == Y)
    return acc
