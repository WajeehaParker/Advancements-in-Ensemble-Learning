import numpy as np
from scipy.stats import mode
from getNNPredict import getNNPredict

def fusionPSO(classifiers, testData):
    tempPredict = []
    index = 0
    X = testData[:, :-1]
    y = testData[:, -1]

    for i in range(len(classifiers)):
        try:
            if classifiers[i]['name'] == 'SVM':
                tempPredict.append(classifiers[i]['model'].predict(X))
                index += 1
            elif classifiers[i]['name'] == 'DT':
                tempPredict.append(classifiers[i]['model'].predict(X))
                index += 1
            elif classifiers[i]['name'] == 'DISCR':
                tempPredict.append(classifiers[i]['model'].predict(X))
                index += 1
            elif classifiers[i]['name'] == 'KNN':
                tempPredict.append(classifiers[i]['model'].predict(X))
                index += 1
            elif classifiers[i]['name'] == 'NB':
                tempPredict.append(classifiers[i]['model'].predict(X))
                index += 1
            elif classifiers[i]['name'] == 'ANN':
                tempPredict.append(getNNPredict(classifiers[i]['model'], X))
                index += 1
        except Exception as ME:
            print('in PSO FUSION')
            continue

    decisionMatrix = np.ones((len(testData[:, 0]), len(tempPredict)))

    for j in range(len(tempPredict)):
        decisionMatrix[:, j] = tempPredict[j]

    fusion = mode(decisionMatrix, axis=1)[0]
    return fusion
