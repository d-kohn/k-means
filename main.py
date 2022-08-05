import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def mse_calc(cluster_x, cluster_y, centroid):
    sum = sum(cluster_x)

datafile = './545_cluster_dataset programming 3.csv'
X = 0
Y = 1
DIM = 2
CLUSTERS = 4
CLUSTER = []
MAX_ITERATIONS = 50
COORD = {X : 'X', Y : 'Y'}
SCALE = 1

# Load data 
with open(datafile, newline='') as csvfile:
    data = pd.read_csv(csvfile, sep='  ', engine='python')
N = len(data)

# Randomly select centroids 
centroid = [[0,0] for i in range(CLUSTERS)]
for c in range(CLUSTERS):
    index = random.randint(0,N)
    centroid[c][X] = data[COORD[X]][index]
    centroid[c][Y] = data[COORD[Y]][index]

# Display full screen
# manager = plt.get_current_fig_manager()
# manager.full_screen_toggle()
# SCALE = 5

done = False
iteration = 0
while (done == False and iteration <= MAX_ITERATIONS):
    done = True
    cluster_assignment = []
    # Create cluster dataframes to store X and
    for cluster in range (CLUSTERS):
        cluster_assignment.append([])
        cluster_assignment[cluster].append(pd.DataFrame({'X':[]}))
        cluster_assignment[cluster].append(pd.DataFrame({'Y':[]}))

    distances = pd.DataFrame()
    for c in range(CLUSTERS):
        centroid_tmp = np.tile(centroid[c], [N,1])
        distances_tmp = pd.DataFrame(np.subtract(data, centroid[c]))
        distances_tmp = distances_tmp.pow(2)
        distances[c] = distances_tmp.sum(axis=1)

    closest_centroid = distances.idxmin(axis=1)
    for index in range(N-1):
        length = len(cluster_assignment[closest_centroid[index]][X].index)
        cluster_assignment[closest_centroid[index]][X].loc[length] = [data[COORD[X]][index]]
        cluster_assignment[closest_centroid[index]][Y].loc[length] = [data[COORD[Y]][index]]
    
    plt.clf()
    for cluster in range(CLUSTERS):
        x = cluster_assignment[cluster][X]
        y = cluster_assignment[cluster][Y]
        plt.scatter(x, y, s=(10 * SCALE))
        plt.scatter(centroid[cluster][X], centroid[cluster][Y], color = 'black', s=(20 * SCALE))
        new_centroid_x = float(cluster_assignment[cluster][X].sum())/len(cluster_assignment[cluster][X].index)
        new_centroid_y = float(cluster_assignment[cluster][Y].sum())/len(cluster_assignment[cluster][Y].index) 
        if (centroid[cluster][X] / new_centroid_x < 0.98 or centroid[cluster][X] / new_centroid_x > 1.02): 
            done = False
        if (centroid[cluster][Y] / new_centroid_y < 0.98 or centroid[cluster][Y] / new_centroid_y > 1.02): 
            done = False
        centroid[cluster][X]= new_centroid_x
        centroid[cluster][Y]= new_centroid_y
    iteration += 1
    print(f'Iteration: {iteration}')
    plt.pause(0.01)
print("FIN")
plt.show()
