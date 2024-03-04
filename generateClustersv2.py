from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def generateClustersv2(train, params):
    print("Generating Clusters")
    totalClusters = 0
    genClusters = []
    noOfClusters = round(np.power(len(train[:, -1]), 1/5))
    
    for clusters in range(1, noOfClusters + 1):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(train[:, :-1])
        kmeans = KMeans(n_clusters=clusters, max_iter=24000).fit(scaled_data)
        
        for j in range(clusters):
            clusterData = train[kmeans.labels_ == j, :]
            genClusters.append(clusterData)
            totalClusters += 1
    
    return genClusters
