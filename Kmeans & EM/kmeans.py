# -*- coding: utf-8 -*-

import argparse
import numpy as np

def kmeans(data, k, maxiter = 50):
    iter = 0
    dist = np.zeros(k)
    n = len(data)
    center = np.zeros(k)
    random_num = np.random.randint(n, size = k)
    # initial the center
    for i in range(k):
        center[i] = data[random_num[i]] 
    
    for i in range(maxiter):
        cluster = [[] for i in range(k)]
        # compute distance
        for i in range(n):
            for j in range(k):
                dist[j] = np.sqrt((data[i] - center[j]) ** 2)
                min_index = np.argmin(dist)
            #step 3: assign point
            cluster[min_index].append(data[i])
        #step 4: update center
        for i in range(k):
            center[i] = np.mean(cluster[i])
        iter += 1
    return center, cluster

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='kmeans')
    parser.add_argument('--data', type=str, dest='filename',
                        help='filename', default='data.csv')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    
    args = parse_args()
    data = args.filename
    data = np.loadtxt(data, delimiter=',')
    k =2
    center, cluster = kmeans(data, k)
    print("center:",center)
    for i in range(k):
        output = 'kmeans' + str(i) + '.csv'
        np.savetxt(output, cluster[i], delimiter=',')
