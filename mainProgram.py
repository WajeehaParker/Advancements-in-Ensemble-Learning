import os
import numpy as np
from multiprocessing import Pool
from functools import partial
from runTraining import runTraining
from saveResults import saveResults

def run_problem(p_name, params):
    results = {}
    nonOptimized_Accuracy = []
    optimized_Accuracy = []
    #for runs in range(params['numOfRuns']):
    print("p_name: "+p_name)
    results = runTraining(p_name, params)
    nonOptimized_Accuracy.append(results['nonOptimized_Accuracy'])
    optimized_Accuracy.append(results['optimized_Accuracy'])

    results['p_name'] = p_name
    results['nonOptimized_Accuracy'] = np.mean(nonOptimized_Accuracy)
    results['optimized_Accuracy'] = np.mean(optimized_Accuracy)
    results['nonOptimized_stdDEV'] = np.std(nonOptimized_Accuracy)
    results['optimized_stdDEV'] = np.std(optimized_Accuracy)
    saveResults(results)

def mainProgram():
    problem = ['breast-cancer-wisconsin', 'ecoli',
               'haberman', 'ionosphere', 'iris', 'liver',
               'pima_diabetec',
               'sonar', 'wine']
    #'diabetic_retinopathy', 'segment2', 'thyroid', 'vehicle'

    # Model SETTINGS
    params = {
        'numOfRuns': 1,
        'numOfFolds': 1,
        'classifiers': ['KNN', 'ANN', 'DT', 'DISCR', 'NB', 'SVM'],
        'trainFunctionANN': ['trainlm', 'trainbfg', 'trainrp', 'trainscg', 'traincgb', 'traincgf', 'traincgp', 'trainoss', 'traingdx'],
        'trainFunctionDiscriminant': ['pseudoLinear', 'pseudoQuadratic'],
        'kernelFunctionSVM': ['gaussian', 'polynomial', 'linear']
    }

    #with Pool(os.cpu_count()) as pool:
    #    pool.map(partial(run_problem, params=params), problem)
    for p_name in problem:
        run_problem(p_name, params)

if __name__ == "__main__":
    mainProgram()
