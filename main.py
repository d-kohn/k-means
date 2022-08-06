import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

datafile = './545_cluster_dataset programming 3.csv'
X = 0
Y = 1
DIM = 2
CLUSTERS = 15
CLUSTER = []
RUNS = 1
MAX_ITERATIONS = 50
COORD = {X : 'X', Y : 'Y'}
SCALE = 1
EQUILIBRIUM = 0.02
FULLSCREEN = True
OUTPUT = False
FONTSIZE = 10
if CLUSTERS > 12:
    FONTSIZE -= int((CLUSTERS - 9)/ 3)
    if FONTSIZE < 5:
        FONTSIZE = 5

def plot_scatter(x, y, centroid):
    # Plot cluster points
    plt.scatter(x, y, s=(10 * SCALE))
    # Plot cluster centroid
    plt.scatter(centroid[X], centroid[Y], color = 'black', s=(40 * SCALE))
    
def calc_mse(clusters, centroid):
    mse = []
    # Calculate Mean Squares Error for each cluster
    for c in range(CLUSTERS):
        # Subtract X and Y values of the centroid from every X and Y value in the cluster
        mse.append(np.subtract(clusters[c], centroid[c]))
        # Square all the X and Y values 
        mse[c] = mse[c].pow(2)
        # Sum the X and Y values for each data point in the cluster (distance)
        mse[c] = mse[c].sum(axis=1)
        # Average the distances from the centroid (MSE)
        mse[c] = np.sum(mse[c]) / len(mse[c])
    return mse

def calc_avg_mse(mse):
    # Find the average MSE
    return sum(mse)/len(mse)

def calc_mss(centroid):
    matrix = []
    # Create a data frame with the centroid data
    df = pd.DataFrame(data = centroid)
    for d in range(DIM):
        matrix.append(np.array(pd.DataFrame(np.array(df[d]))))
        vector = np.transpose(matrix[d])
        matrix[d] = np.tile(matrix[d], (1, CLUSTERS))
        matrix[d] = np.subtract(matrix[d], vector)
        matrix[d] = np.power(matrix[d], 2)
    sum = np.sum(matrix) / 2
    mss = (sum) / (CLUSTERS * (CLUSTERS-1) / 2)
    return mss

# Load data 
with open(datafile, newline='') as csvfile:
    data = pd.read_csv(csvfile, sep='  ', engine='python')
N = len(data)

# Display full screen
if (FULLSCREEN == True):
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    SCALE = 5

for run in range(RUNS):
    # Randomly select centroids 
    centroid = [[0,0] for i in range(CLUSTERS)]
    for c in range(CLUSTERS):
        index = random.randint(0,N-1)
        centroid[c][X] = data[COORD[X]][index]
        centroid[c][Y] = data[COORD[Y]][index]

    done = False
    iteration = 1
    while (done == False and iteration <= MAX_ITERATIONS):
        print(f'Iteration: {iteration}')
        done = True
        distances = pd.DataFrame()
        # Calculate distances from each point to every centroid and store in 'distances'
        for c in range(CLUSTERS):
            distances_tmp = pd.DataFrame(np.subtract(data, centroid[c]))
            distances_tmp = distances_tmp.pow(2)
            distances[c] = distances_tmp.sum(axis=1)
        # Find the closest centroid for each point, store result in 'closest_centroid'
        closest_centroid = distances.idxmin(axis=1)
 
        cluster_assignment = []
        cluster_size = []
        # Create cluster dataframes to store X and Y coords
        for cluster in range (CLUSTERS):
            cluster_assignment.append(pd.DataFrame(columns = ['X', 'Y']))
            cluster_size.append(0)
        # Sort data into their clusters
        for index in range(N-1):
            # Get the cluster closest to the current data index
            cluster = closest_centroid[index]
            # Assign the current data to it's cluster
            cluster_assignment[cluster].loc[cluster_size[cluster]] = data.iloc[index]
            # Increment the size of the current cluster
            cluster_size[cluster] += 1

        # Calculate Mean Square Error,Average MSE,Mean Square Separation, and Entropy
        mse = calc_mse(cluster_assignment, centroid)
        avg_mse = calc_avg_mse(mse)
        mss = calc_mss(centroid)

        plt.clf()
        for cluster in range(CLUSTERS):
            x = cluster_assignment[cluster][COORD[X]]
            y = cluster_assignment[cluster][COORD[Y]]
            # Plot cluster points
            plot_scatter(x,y,centroid[cluster])
            # Calc new centroids as mean of all X and Y values in the cluster
            new_centroid_x = float(x.sum())/len(x.index)
            new_centroid_y = float(y.sum())/len(y.index) 
            # Check - Change in X and Y value < EQUILIBRIUM? 'done' when all centroids X & Y DELTA < EQUILIBRIUM
            if (np.abs((centroid[cluster][X] / new_centroid_x) - 1) > EQUILIBRIUM): 
                done = False
            if (np.abs((centroid[cluster][Y] / new_centroid_y) - 1) > EQUILIBRIUM): 
                done = False
            # Store new centroids
            centroid[cluster][X]= new_centroid_x
            centroid[cluster][Y]= new_centroid_y
        # Display average MSE 
        plt.text(-5,-5.2, f'Iteration: {iteration}   AVG MSE: {round(avg_mse, 3)}   MSS: {round(mss, 3)}')
        # Display cluster MSEs
        for cluster in range(CLUSTERS):
            x_pos = -5 + (.17*FONTSIZE * int(cluster / 3))
            y_pos = 5.2 - (0.3 * (cluster % 3))
            plt.text(x_pos, y_pos,f'MSE {cluster}: {round(mse[cluster],SCALE+1)}', fontsize=FONTSIZE+SCALE)
        plt.pause(0.01)
        if(OUTPUT == True):
            plt.savefig(f'output-c{CLUSTERS}-r{run}-i{iteration}.jpg', fontsize=9+SCALE)
        iteration += 1
    print("FIN")
    plt.show()

