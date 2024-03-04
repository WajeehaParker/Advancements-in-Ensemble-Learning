from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from trainNN import trainNN
#from trainRBFNN import trainRBFNN
from trainSVM import trainSVM

def trainClassifiers(X, y, valX, valy, params):
    classifiers = []
    index = 0
    
    for learner in params['classifiers']:
        try:
            if learner == 'KNN':
                model = KNeighborsClassifier(n_neighbors=params['n_neighbors'])
                model.fit(X, y)
                classifiers.append({'name': 'KNN', 'model': model})
                index += 1
            elif learner == 'SVM':
                model = trainSVM(X, y, valX, valy)
                classifiers.append(model)
                index += 1
            elif learner == 'NB':
                model = GaussianNB()
                model.fit(X, y)
                classifiers.append({'name': 'NB', 'model': model})
                index += 1
            elif learner == 'DISCR':
                model = LinearDiscriminantAnalysis()
                model.fit(X, y)
                classifiers.append({'name': 'DISCR', 'model': model})
                index += 1
            elif learner == 'DT':
                model = DecisionTreeClassifier()
                model.fit(X, y)
                classifiers.append({'name': 'DT', 'model': model})
                index += 1
            elif learner == 'ANN':
                print("Inside ANN case")
                model = trainNN(X, y, params, p)
                classifiers.append({'name': 'ANN', 'model': model})
                index += 1
            #elif learner == 'RBFNN':
             #   model = trainRBFNN(X, y, valX, valy)
              #  classifiers.append(model)
               # index += 1
            #elif learner == 'LPBOOST':
            elif learner == 'BAG':
                model = BaggingClassifier()  # Implement Bagging Classifier
                model.fit(X, y)
                classifiers.append({'name': 'BAG', 'model': model})
                index += 1
            #elif learner == 'SUBSPACE':
            #elif learner == 'TOTALBOOST':
            elif learner == 'ADABOOST':
                model = AdaBoostClassifier()  # Implement AdaBoost Classifier
                model.fit(X, y)
                classifiers.append({'name': 'ADABOOST', 'model': model})
                index += 1
            else:
                print('Unknown Classifier')
        except Exception as exc:
            print(f'Something happened in trainClassifiers: {exc}')
    
    return classifiers
