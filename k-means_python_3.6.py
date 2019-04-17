import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy


"""
################################################################
############# PLOTING POINTS AND INITIAL CENTEROID #############
################################################################
"""
x_coord = np.array([1,1,2,2,3,4,5,6])
y_coord = np.array([1,2,1,3,3,5,4,5])

x_centroid = np.array([2.0,3.0,5.0])
y_centroid = np.array([3.0,3.0,4.0])

plt.figure(figsize=(5, 5))
plt.scatter(x_coord, y_coord, color='k')

color_array = np.array(['r','b','g'])
for i in range(len(x_centroid)):
    plt.scatter(x_centroid[i], y_centroid[i] ,color = color_array[i])

plt.xlim(0, 7)
plt.ylim(0, 7)
plt.xlabel('X', fontsize='13', color='r')
plt.ylabel('Y', fontsize='13', color='r')
plt.show()


"""
################################################################
################### EUCLIDEAN DISTANCE FROM CENTROID ###########
################################################################
"""

def distance(x_coord, y_coord, x_centroid, y_centroid):    
#    print('EUCLIDEAN DISTANCE FROM CENTROID :\n')
    dist_mat = np.zeros(shape=(len(x_coord),len(x_centroid)))
    for i in range(len(x_centroid)):
        for j in range(len(x_coord)):
            dist_mat[j,i] = np.sqrt((x_centroid[i] - x_coord[j])**2 + (y_centroid[i] - y_coord[j])**2)
    
    dataframe = pd.DataFrame(dist_mat)
    
    distance_columns = dataframe.columns =  ['c1','c2', 'c3']
    dataframe['cluster'] = dataframe.loc[:, distance_columns].idxmin(axis=1)
   
    for i in range(len(color_array)):
        dataframe.loc[dataframe["cluster"] == 'c{}'.format(i+1), "color"] = color_array[i]

    dataframe['x'] = x_coord
    dataframe['y'] = y_coord
    
    print(dataframe[['x','y', 'cluster']])  
   
    plt.figure(figsize=(5, 5))
    plt.scatter(x_coord, y_coord, color=dataframe['color'], alpha=0.5)
#    for i in range(len(x_centroid)):
#        plt.scatter(x_centroid[i], y_centroid[i] ,color = color_array[i])
    
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    plt.show()
    return dataframe


"""
################################################################
################### SUM OF SQUARED ERROR #######################
################################################################
"""


def ss_error():
    sum_error = 0.00
    for i in range(len(x_centroid)):
        cluster_size = len(dataframe[dataframe["cluster"] == 'c{}'.format(i+1)])
        x_temp = dataframe.loc[dataframe["cluster"] == 'c{}'.format(i+1)]['x'].values
        y_temp = dataframe.loc[dataframe["cluster"] == 'c{}'.format(i+1)]['y'].values
        for j in range(cluster_size):
            sum_error = sum_error + ((x_temp[j] - x_centroid[i]) **2 + (y_temp[j] - y_centroid[i])**2)
    print('\nSum of Squared Error : ', "%0.2f" %sum_error,'\n')


"""
################################################################
#################### CALCULATE NEW CENTROID ####################
################################################################
"""

def updated_centroid(x_centroid,y_centroid):
    print('Updated Centroid :','\n')
    print('index', '\t ', 'x_c', '\t ', 'y_c', '\n')    

    for i in range(len(x_centroid)):
        x_centroid[i] = np.mean(dataframe.loc[dataframe["cluster"] == 'c{}'.format(i+1)]['x'])
        y_centroid[i] = np.mean(dataframe.loc[dataframe["cluster"] == 'c{}'.format(i+1)]['y'])        
    
        print(i, '\t ', "%0.2f" %x_centroid[i], '\t ', "%0.2f" % y_centroid[i], '\n')    

    print('\n\n')

"""
################################################################
######################### K-MEANS ITERATION ####################
################################################################
"""

while True:
    
    dataframe = distance(x_coord, y_coord, x_centroid, y_centroid)
    ss_error()
    old_x_centroids = copy.deepcopy(x_centroid)
    old_y_centroids = copy.deepcopy(y_centroid)
    updated_centroid(x_centroid, y_centroid)    
    if (np.array_equal(old_x_centroids, x_centroid) and np.array_equal(old_y_centroids, y_centroid)):
        break

plt.figure(figsize=(5, 5))
plt.scatter(x_coord, y_coord, color=dataframe['color'], alpha=0.5)
plt.xlim(0, 7)
plt.ylim(0, 7)
plt.xlabel('X', fontsize='13', color='r')
plt.ylabel('Y', fontsize='13', color='r')
plt.show()
