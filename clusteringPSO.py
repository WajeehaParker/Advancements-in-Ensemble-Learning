import numpy as np
from scipy.stats import mode
from pyswarm import pso
from trainClassifiers import trainClassifiers
from fusionPSO import fusionPSO

def clusteringPSO(allClusters, testData, params):
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
        allPredictions = np.zeros((len(testData[:, -1]), len(allClusters)))
        
        for j in range(len(allClusters)):
            all = []
            params['classifiers'] = ['SVM']
            all = trainClassifiers(allClusters[j][:, :-1], allClusters[j][:, -1], testData[:, :-1], testData[:, -1], params)
            allPredictions[:, j] = fusionPSO(all, testData)

        lb = np.zeros(len(allPredictions))
        ub = np.ones(len(allPredictions))
        best, fval = pso(PSOAF, lb, ub, swarmsize=50)
        
        obj = {
            'chromosome': np.round(best),
            'fval': fval
        }
    except Exception as exc:
        print(f'Problem with {exc}')
        obj = None

    return obj
