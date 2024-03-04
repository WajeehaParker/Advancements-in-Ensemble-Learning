import csv
import os

def saveResults(results):
    file_path = 'results.csv'
    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Data Set', 'Avg Accuracy', 'Std. Dev', 'Optimized Acc', 'Std. Dev'])

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            results['p_name'],
            results['nonOptimized_Accuracy'],
            results['nonOptimized_stdDEV'],
            results['optimized_Accuracy'],
            results['optimized_stdDEV']
        ])
