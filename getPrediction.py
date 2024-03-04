from getNNPredict import getNNPredict

def getPrediction(classifier, X):
    try:
        if classifier['name'] == 'SVM':
            preds = classifier['model'].predict(X)
        elif classifier['name'] == 'KNN':
            preds = classifier['model'].predict(X)
        elif classifier['name'] == 'DT':
            preds = classifier['model'].predict(X)
        elif classifier['name'] == 'NB':
            preds = classifier['model'].predict(X)
        elif classifier['name'] == 'DISCR':
            preds = classifier['model'].predict(X)
        elif classifier['name'] == 'ANN':
            preds = getNNPredict(classifier['model'], X)
        elif classifier['name'] == 'CNN':
            preds = getCNNPred(classifier['model'], X)
        elif classifier['name'] == 'RBFNN':
            preds = getNNPredict(classifier['model'], X)
    except Exception as ME:
        print(f'Problem in getPrediction: {ME}')
    
    return preds
