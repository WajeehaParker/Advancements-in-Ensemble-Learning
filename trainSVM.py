from sklearn.svm import SVC

def trainSVM(X, y, valX, valy):
    # Create an SVM classifier object
    svm_model = SVC(kernel='linear')  # You can specify different kernels if needed
    
    # Train the SVM classifier
    svm_model.fit(X, y)
    
    return {'name': 'SVM', 'model': svm_model}
