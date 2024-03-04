import numpy as np
from scipy.stats import mode
from pyswarm import pso
from psoPredict import psoPredict

def classifierSelectionPSO(classifierList, testData):
    def PSOAF(c):
        c = c > 0.6
        c = np.where(c)[0]
        
        decisionMatrix = np.ones((len(testData[:, -1]), len(c)))
        for i in range(len(c)):
            decisionMatrix[:, i] = allPredictions[:, c[i]]
        
        decisionMatrix = mode(decisionMatrix, axis=1)[0]
        error = np.mean(decisionMatrix != testData[:, -1])
        return error

    try:
        allPredictions = psoPredict(classifierList, testData)
        lb = np.zeros(len(classifierList))
        ub = np.ones(len(classifierList))
        best, fval = pso(PSOAF, lb, ub, swarmsize=50)
        
        obj = {
            'chromosome': np.round(best),
            'fval': fval
        }
    except Exception as exc:
        print(f'Problem with {exc}')
        obj = None

    return obj
