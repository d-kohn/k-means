import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

datafile = './545_cluster_dataset programming 3.csv'
X = 0
Y = 1
DIM = 2
CLUSTERS = 5
CLUSTER = []
ITERATIONS = 10
COORD = {X : 'X', Y : 'Y'}
COLORS = {0 : "red", 1 : "blue", 2 : "green", 3 : "yellow", 4 : "brown"}
for i in range(CLUSTERS):
    CLUSTER.append(i)

centroid = [[0,0] for i in range(CLUSTERS)]

# Load data 
with open(datafile, newline='') as csvfile:
    data = pd.read_csv(csvfile, sep='  ', engine='python')
N = len(data)

# Randomly select centroids 
for c in range(CLUSTERS):
    index = random.randint(0,N)
    centroid[c][X] = data[COORD[X]][index]
    centroid[c][Y] = data[COORD[Y]][index]

manager = plt.get_current_fig_manager()
manager.full_screen_toggle()
for i in range(ITERATIONS):
    cluster_assignment = []
    for c in range (CLUSTERS):
        cluster_assignment.append([])
        cluster_assignment[CLUSTER[c]].append(pd.DataFrame({'X':[]}))
        cluster_assignment[CLUSTER[c]].append(pd.DataFrame({'Y':[]}))

    distances = pd.DataFrame()
    for c in range(CLUSTERS):
        centroid_tmp = np.tile(centroid[c], [N,1])
        distances_tmp = pd.DataFrame(np.subtract(data, centroid[c]))
        distances_tmp = distances_tmp.pow(2)
        distances[c] = distances_tmp.sum(axis=1)

    distances = np.transpose(distances)
    for index in range(N-1):
        closest_centroid = distances[index].idxmin()
        length = len(cluster_assignment[closest_centroid][X].index)
        cluster_assignment[closest_centroid][X].loc[length] = [data[COORD[X]][index]]
        cluster_assignment[closest_centroid][Y].loc[length] = [data[COORD[Y]][index]]
    
    plt.clf()
    for cluster in range(CLUSTERS):
        x = cluster_assignment[cluster][X]
        y = cluster_assignment[cluster][Y]
        plt.scatter(x, y, color = COLORS[cluster], s=50)
        plt.scatter(centroid[cluster][X], centroid[cluster][Y], color = 'black', s=200)
        centroid[cluster][X]= float(cluster_assignment[cluster][X].sum())/len(cluster_assignment[cluster][X].index)
        centroid[cluster][Y]= float(cluster_assignment[cluster][Y].sum())/len(cluster_assignment[cluster][Y].index)

    plt.pause(0.01)
plt.show()