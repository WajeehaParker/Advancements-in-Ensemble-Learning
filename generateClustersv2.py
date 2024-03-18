from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def generateClustersv2(train, params):
    totalClusters = 0
    genClusters = []
    noOfClusters = round(np.power(len(train[:, -1]), 1/5)) 
    print("Generated Clusters : ", noOfClusters)
    for clusters in range(1, noOfClusters + 1):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(train[:, :-1])
        kmeans = KMeans(n_clusters=clusters, max_iter=24000).fit(scaled_data)
        
        for j in range(clusters):
            clusterData = train[kmeans.labels_ == j, :]

            # Check if all y values in the cluster are identical
            unique_y_values = np.unique(clusterData[:, -1])
            if len(unique_y_values) == 1:
                print(f"Skipping cluster {j+1} as all y values are identical.")
                continue

            genClusters.append(clusterData)
            totalClusters += 1
    
    return genClusters
