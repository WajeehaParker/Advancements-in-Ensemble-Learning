import os
import numpy as np
from generateClustersv2 import generateClustersv2
from clusteringPSO import clusteringPSO
from trainClassifiers import trainClassifiers
from classifierSelectionPSO import classifierSelectionPSO
from fusion import fusion
from sklearn.model_selection import train_test_split

def runTraining(p_name, params):
    results = {}
    nonOptimized_Accuracy = []
    optimized_Accuracy = []
    classifiers = []

    '''
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

    ## SEPARATE VALIDATION DATA
    cvv = np.random.rand(len(trainData)) < 0.9
    valData = trainData[~cvv]
    trainData = trainData[cvv]

    trainX = trainData[:, :-1]
    trainy = trainData[:, -1]

    valX = valData[:, :-1]
    valy = valData[:, -1]
    '''
    ''''''
    file_path = os.path.join('DTE', p_name, 'data.csv')
    data = np.genfromtxt(file_path, delimiter=',', skip_header=1)  # Adjust delimiter if necessary

    # Split the data into features (X) and labels (Y)
    X = data[:, :-1]
    Y = data[:, -1]

    # Split the data into training and testing sets
    X, X_test, Y, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)  # Adjust test_size if needed
    # Further split the training data into training and validation sets
    trainX, valX, trainy, valy = train_test_split(X, Y, test_size=0.25, random_state=42)  # Adjust test_size if needed
    ''''''
    
    print("Data Loaded Successfully")

    allClusters = generateClustersv2(np.column_stack((trainX, trainy)), params)
    bestClusters = clusteringPSO(allClusters, np.column_stack((valX, valy)), params)
    bestClusters = np.flatnonzero(bestClusters['chromosome'])
    print("bestClusters: ",bestClusters)
    selectedClusters = [allClusters[i] for i in bestClusters]

    for c in selectedClusters:
        X = c[:, :-1]
        y = c[:, -1]
        all = trainClassifiers(X, y, valX, valy, params)
        classifiers.extend(all)

    print("classifiers: ",classifiers)
    psoEnsemble = classifierSelectionPSO(classifiers, np.column_stack((valX, valy)))
    print("psoEnsemble: ",psoEnsemble)
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
