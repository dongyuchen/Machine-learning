# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 00:24:30 2019

@author: 37112
"""
import sys
import numpy as np
import copy
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def data_generate(a,b,c,d,e,f):
    n1 = a
    n2 = b
    μ1 = c
    μ2 = d
    sigma1 = e
    sigma2 = f
    data1 = np.zeros(shape=(n1,2),dtype=int)
    data2 = np.zeros(shape=(n2,2),dtype=int)
#    np.random.seed(0)
    s1 = np.random.normal(μ1, sigma1, n1)
    s1 = s1.astype(int)
    for i in range(len(s1)):
        data1[i][0] = s1[i]
        data1[i][1] = 0
    s2 = np.random.normal(μ2, sigma2, n2)
    s2 = s2.astype(int)
    for i in range(len(s2)):
        data2[i][0] = s2[i]
        data2[i][1] = 1
    data = np.zeros(shape=(n1+n2,2),dtype=int)
    for i in range(n1+n2):
        if i >= n1:
            data[i] = data2[i-n1]
        else:
            data[i] = data1[i]
    np.random.shuffle(data)
    return data

def init_kmeans(data, k):
    centers = np.zeros(k)
    for i in range(k):
        index = int(np.random.uniform(0, len(data)))
        centers[i] = data[index]
    return centers

def init_em(k):
    weight = np.zeros(k)
    μ = np.zeros(k)
    sigma = np.zeros(k)
    n = 1
    a = 20
    for i in range(k):
        if i == k-1:
            weight[i] = n
        else:
            weight[i] = np.random.uniform(0,n)
        n -= weight[i]
    for i in range(k):
        μ[i] = np.random.randint(a)
    for i in range(k):
        sigma[i] = np.random.randint(a)
        if sigma[i] == 0:
           sigma[i] += 1 
    return weight,μ,sigma

def com_SSE(cluster, centers):
    SSE = 0
    for i in range(len(cluster)):
        for j in range(len(cluster[i])):
           SSE += np.power(cluster[i][j] - centers[i], 2)

    return SSE

def kmeans(data, k, threshold = 1e-5, maxiter = 200):
    iter = 0
    distance = 0
    cluster = np.zeros(shape =(len(data),2), dtype=int)
    for i in range(len(data)):
        cluster[i][0] = data[i]
    # step 1: initialization
    centers = init_kmeans(data, k)
    centers_prev = copy.deepcopy(centers)
    while True:
        # step 2: distance computation
        for i in range(len(data)):
            min_dist = 100000
            min_label = 0
            for j in range(k):
                # Because the data is one dim, so euclidean metric
                # is equal to Manhattan distance
                distance = np.sqrt((data[i] - centers[j]) ** 2)
#                distance = np.abs(data[i] - centers[j])
                if distance < min_dist:
                    min_dist  = distance
                    min_label = j
            #step 3: assignment of a data point to a cluster
            if cluster[i][1] != min_label:
                cluster[i][1] = min_label
        #step 4: update centroids
        for i in range(k):
            points = np.extract(cluster[:, 1] == i, cluster[:,0])
            centers[i] = np.mean(points)
        error = np.abs(np.mean(centers_prev - centers))
        if error < threshold:
            print("Kmeans method uses {} iter".format(iter))
            break
        if iter >= maxiter:
            break
        iter += 1
    return centers, cluster

def em(data, k, threshold = 1e-6, maxiter = 200):
    iter = 0
    n = len(data)
    cluster = np.zeros(shape =(len(data),2), dtype=int)
    prob = np.zeros(k)
    posterior = np.zeros(shape=(n,k))
    # step 1: initialization
    weight,μ,sigma  = init_em(k)
    # step 2: (E step)
    # computing the posterior probability
    while True:
        μ_prev = copy.deepcopy(μ)
        weight_prev = copy.deepcopy(weight)
        sigma_prev = copy.deepcopy(sigma)
        for i in range(n):
            sum_prob = 0
            for j in range(k):
                if (2 * np.pi * sigma[j]) == 0:
                    print("sigma[j]:",sigma[j])
                prob[j] = weight[j] * (1.0/(np.sqrt(2 * np.pi * sigma[j])) * np.exp(-1.0/(2.0 * sigma[j]) * (data[i] - μ[j])**2))
                if prob[j] == 0:
                    prob[j] += 1e-200
                sum_prob += prob[j]
#                if sum_prob == 0:
#                    print("data[i]:{},μ:{},sigma:{},prob:{},i:{},iter:{}".format(data[i],μ,sigma,prob,i,iter))
            
            posterior[i] = prob / sum_prob
                
        # step 3: updating the weight, μ and sigma of the normal 
        # distributions (M step)
        for j in range(k):
            total_exp = 0
            total_post = 0
            var = 0
            for i in range(n):
                total_exp += posterior[i,j] * data[i]
                total_post += posterior[i,j]
            μ[j] = total_exp/total_post
            weight[j] = total_post/n
            for i in range(n):
                var += posterior[i,j] * (data[i] - μ[j])**2
#            sigma[j] = np.sqrt(var/total_post)
            sigma[j] = var/total_post

        prediction = [np.argmax(posterior[i]) for i in range(n)]
        for i in range(n):
            cluster[i][0] = data[i]
            cluster[i][1] = prediction[i]
        error = np.sum(np.abs(μ_prev - μ)) + np.sum(np.abs(weight_prev - weight)) + \
                np.sum(np.abs(sigma_prev - sigma))

        if error < threshold:
            print("EM method uses {} iter".format(iter))
            break
        if iter >= maxiter:
            break
        iter += 1
#        print("iter:",iter)
#        print("posterior:")
#        print(posterior)
#        print("weight:",weight)
#        print("mean:",μ)
#        print("standard deviation",np.sqrt(sigma))
    return weight, μ, sigma, cluster

def acu_curve(y,prob):
    fpr,tpr,threshold = roc_curve(y,prob) 
    roc_auc = auc(fpr,tpr) ###auc
 
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) 
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
 
    plt.show()

def cal_acc(data, cluster):
    n = len(data)
    num = 0
    for i in range(n):
        if data[i][1] == cluster[i][1]:
            num += 1
    accuracy = num / n
    if accuracy < 0.5:
        accuracy = 1 - accuracy
    return accuracy

if __name__ == '__main__':
    # data just for one dim
    n = len(sys.argv)
    if n == 1:
        print("Please enter the data which need to be test")
    elif n == 2:
        data = np.loadtxt(sys.argv[1], delimiter=',')
    else:
        data = []
        for i in range(1,n):
            data.append(int(sys.argv[i]))

#    data = np.loadtxt('data.csv', delimiter=',')
#    data = data.reshape(len(data),-1)
#    print(data)
#    data = data_generate(5000,1000,50,5,5,10)
#    print(data)
    k = 2
    cluster_arr = np.zeros(k,dtype = int)
    # kmeans method
    kmeans_time = time.time()
    centers, kmeans_cluster = kmeans(data, k)
    for i in range(k):
        cluster_arr = np.extract(kmeans_cluster[:, 1] == i, kmeans_cluster[:,0])
        print("kmeans std:", np.std(cluster_arr))
#        print("kmeans cluster_{} :{}".format(i, cluster_arr))
        filename = 'kmeans_cluster' + str(i) + '.csv'
        np.savetxt(filename, cluster_arr, delimiter=',')
    print("kmeans centers:",centers)
#    accuracy = cal_acc(data, kmeans_cluster)
#    print("kmeans accuracy:", accuracy)
    
#    acu_curve(data[:,1],kmeans_cluster[:,1]) #roc
    print("kmeans time:", time.time() - kmeans_time)
    # EM alogrithom
    em_time = time.time()
    weight, μ, sigma, em_cluster = em(data,k)
    for i in range(k):
        cluster_arr = np.extract(em_cluster[:, 1] == i, em_cluster[:,0])
#        print("EM cluster_{} :{}".format(i, cluster_arr))
        filename = 'em_cluster' + str(i) + '.csv'
        np.savetxt(filename, cluster_arr, delimiter=',')
    print("EM method's weight:{}, μ:{}, sigma:{}".format(weight, μ, np.sqrt(sigma)))
#    accuracy1 = cal_acc(data, em_cluster)
#    print("em accuracy:", accuracy1)
#    acu_curve(data[:,1],em_cluster[:,1]) #roc
    print("EM time:", time.time() - em_time)
    