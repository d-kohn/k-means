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
RUNS = 10
MAX_ITERATIONS = 100
COORD = {X : 'X', Y : 'Y'}
SCALE = 1
EQUILIBRIUM = 0.02
FULLSCREEN = False
OUTPUT = True

def plot_scatter(x, y, centroid):
    # Plot cluster points
    plt.scatter(x, y, s=(10 * SCALE))
    # Plot cluster centroid
    plt.scatter(centroid[X], centroid[Y], color = 'black', s=(40 * SCALE))

def calc_mse(clusters, centroid):
    mses = []
    for c in range(CLUSTERS):
        mses.append(np.subtract(clusters[c], centroid[c]))
        mses[c] = mses[c].pow(2)
        mses[c] = mses[c].sum(axis=1)
        mses[c] = mses[c].sum() / len(mses[c])
    return mses

def calc_avg_mse(mses):
    return sum(mses)/len(mses)

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
        index = random.randint(0,N)
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
    # Calc new centroids as mean of all X and Y values in the cluster

        cluster_assignment = []
        # Create cluster dataframes to store X and Y coords
        for cluster in range (CLUSTERS):
            cluster_assignment.append(pd.DataFrame(columns = ['X', 'Y']))
        # Sort data into their clusters
        for index in range(N-1):
            length = len(cluster_assignment[closest_centroid[index]][COORD[X]].index)
            cluster_assignment[closest_centroid[index]].loc[length] = data.iloc[index]

        # Calculate Mean Square Error and Average MSE
        mses = calc_mse(cluster_assignment, centroid)
        avg_mse = calc_avg_mse(mses)

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
        plt.text(-5,-5.2, f'AVG MSE: {round(avg_mse, 4)}')
        plt.pause(0.01)
        if(OUTPUT == True):
            plt.savefig(f'output-c{CLUSTERS}-r{run}-i{iteration}.jpg')
        iteration += 1
    print("FIN")
#    plt.show()

aimport random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

datafile = './545_cluster_dataset programming 3.csv'
X = 0
Y = 1
DIM = 2
CLUSTERS = 9
CLUSTER = []
RUNS = 1
MAX_ITERATIONS = 10
COORD = {X : 'X', Y : 'Y'}
SCALE = 1
EQUILIBRIUM = 0.02
FULLSCREEN = False
OUTPUT = False

def plot_scatter(x, y, centroid, ax):
    # Plot cluster points
    plt.scatter(x, y, s=(10 * SCALE))
    # Plot cluster centroid
    plt.scatter(centroid[X], centroid[Y], color = 'black', s=(40 * SCALE))
    
#def plot_info(fig, ax, avg_mse):
    

def calc_mse(clusters, centroid):
    mses = []
    for c in range(CLUSTERS):
        mses.append(np.subtract(clusters[c], centroid[c]))
        mses[c] = mses[c].pow(2)
        mses[c] = mses[c].sum(axis=1)
        mses[c] = mses[c].sum() / len(mses[c])
    return mses

def calc_avg_mse(mses):
    return sum(mses)/len(mses)

# Load data 
with open(datafile, newline='') as csvfile:
    data = pd.read_csv(csvfile, sep='  ', engine='python')
N = len(data)

# Display full screen
if (FULLSCREEN == True):
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    SCALE = 5

fig,ax = plt.subplots()

for run in range(RUNS):
    # Randomly select centroids 
    centroid = [[0,0] for i in range(CLUSTERS)]
    for c in range(CLUSTERS):
        index = random.randint(0,N)
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

        # Calculate Mean Square Error and Average MSE
        mses = calc_mse(cluster_assignment, centroid)
        avg_mse = calc_avg_mse(mses)

        plt.clf()
        for cluster in range(CLUSTERS):
            x = cluster_assignment[cluster][COORD[X]]
            y = cluster_assignment[cluster][COORD[Y]]
            # Plot cluster points
            plot_scatter(x,y,centroid[cluster], ax)
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
        plt.text(-5,-5.2, f'AVG MSE: {round(avg_mse, 4)}')
        plt.pause(0.01)
        if(OUTPUT == True):
            plt.savefig(f'output-c{CLUSTERS}-r{run}-i{iteration}.jpg')
        iteration += 1
    print("FIN")
#    plt.show()

