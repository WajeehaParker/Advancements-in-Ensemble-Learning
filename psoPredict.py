from getNNPredict import getNNPredict

def psoPredict(classifiers, testData):
    X = testData[:, :-1]
    predictions = np.ones((len(testData[:, -1]), len(classifiers)))
    
    for i in range(len(classifiers)):
        try:
            if classifiers[i].name != 'ANN':
                predictions[:, i] = classifiers[i].model.predict(X)
            elif classifiers[i].name == 'ANN':
                predictions[:, i] = getNNPredict(classifiers[i].model, X)
        except Exception as ME:
            print(f'IN psoPredict: {str(ME)}')
            continue
    
    return predictions
