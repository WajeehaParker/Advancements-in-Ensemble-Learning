import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.stats import mode
from getNNPredict import getNNPredict

def accuracyOfPSO(classifiers, chromosome, testData):
    c = np.where(chromosome)[0]
    X = testData[:, :-1]
    y = testData[:, -1]
    decisionMatrix = np.ones((len(testData[:,0]), len(c)))
    
    for i in range(len(c)):
        try:
            if classifiers[i].name == 'SVM':
                decisionMatrix[:, i] = classifiers[i].model.predict(X)
            elif classifiers[i].name == 'KNN':
                decisionMatrix[:, i] = classifiers[i].model.predict(X)
            elif classifiers[i].name == 'DT':
                decisionMatrix[:, i] = classifiers[i].model.predict(X)
            elif classifiers[i].name == 'NB':
                decisionMatrix[:, i] = classifiers[i].model.predict(X)
            elif classifiers[i].name == 'DISCR':
                decisionMatrix[:, i] = classifiers[i].model.predict(X)
            elif classifiers[i].name == 'ANN':
                decisionMatrix[:, i] = getNNPredict(classifiers[i].model, X)
        except Exception as e:
            print(f'IN ACCURACY OF PSO: {str(e)}')
            continue
    
    decisionMatrix = mode(decisionMatrix, axis=1)[0]
    accuracy = accuracy_score(y, decisionMatrix)
    fMeasures = confusion_matrix(y, decisionMatrix)
    
    return accuracy, fMeasures, decisionMatrix
