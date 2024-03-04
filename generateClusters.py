import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def generateClusters(train, params):
    totalClusters = 0
    genClusters = []
    dataClasses = np.unique(train[:, -1])
    
    for dataClass in dataClasses:
        classData = train[train[:, -1] == dataClass, :-1]
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(classData)
        kmeans = KMeans(n_clusters=params['eachClass']).fit(scaled_data)
        
        for j in range(params['eachClass']):
            clusterData = train[kmeans.labels_ == j, :]
            centroid = kmeans.cluster_centers_[j]
            genClusters.append({'train': clusterData, 'centroid': centroid})
            totalClusters += 1
    
    return genClusters
