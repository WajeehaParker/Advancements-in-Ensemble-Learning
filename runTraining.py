import os
import numpy as np
from generateClustersv2 import generateClustersv2
from clusteringPSO import clusteringPSO
from trainClassifiers import trainClassifiers
from classifierSelectionPSO import classifierSelectionPSO
from fusion import fusion

def runTraining(p_name, params):
    results = {}
    nonOptimized_Accuracy = []
    optimized_Accuracy = []
    classifiers = []

    #for f in range(1, params['numOfFolds'] + 1):
    # Load train data
    train_file_path = os.path.join('DTE', p_name, 'train.csv')
    trainData = np.genfromtxt(train_file_path, delimiter=',', skip_header=1)  # Adjust delimiter if necessary

    X = trainData[:, :-1]
    Y = trainData[:, -1]

    # Load test data
    test_file_path = os.path.join('DTE', p_name, 'test.csv')
    testdata = np.genfromtxt(test_file_path, delimiter=',', skip_header=1)  # Adjust delimiter if necessary

    X_test = testdata[:, :-1]
    y_test = testdata[:, -1]

    print("Data Loaded Successfully")

    ## SEPARATE VALIDATION DATA
    cvv = np.random.rand(len(trainData)) < 0.9
    valData = trainData[~cvv]
    trainData = trainData[cvv]

    trainX = trainData[:, :-1]
    trainy = trainData[:, -1]

    valX = valData[:, :-1]
    valy = valData[:, -1]

    allClusters = generateClustersv2(np.column_stack((trainX, trainy)), params)

    bestClusters = clusteringPSO(allClusters, np.column_stack((valX, valy)), params)
    bestClusters = np.flatnonzero(bestClusters['chromosome'])
    selectedClusters = [allClusters[i] for i in bestClusters]

    for c in selectedClusters:
        X = c[:, :-1]
        y = c[:, -1]
        all = trainClassifiers(X, y, valX, valy, params)
        classifiers.extend(all)

    psoEnsemble = classifierSelectionPSO(classifiers, np.column_stack((valX, valy)))
    psoEnsemble = np.flatnonzero(psoEnsemble['chromosome'])
    selectedClassifiers = [classifiers[i] for i in psoEnsemble]

    nonOptimized_Accuracy.append(fusion(classifiers, np.column_stack((X_test, y_test))))
    optimized_Accuracy.append(fusion(selectedClassifiers, np.column_stack((X_test, y_test))))
    #end for

    results['nonOptimized_Accuracy'] = np.mean(nonOptimized_Accuracy)
    results['nonOptimized_stdDEV'] = np.std(nonOptimized_Accuracy)
    results['optimized_Accuracy'] = np.mean(optimized_Accuracy)
    results['optimized_stdDEV'] = np.std(optimized_Accuracy)
    
    return results
