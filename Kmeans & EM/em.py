# -*- coding: utf-8 -*-

import sys
import numpy as np

def em(data, k, threshold = 0.01):
    n = len(data)
    # initial 
    weight = np.random.randint(n, size = k)
    weight = weight/np.sum(weight)
    miu = np.random.rand(k)
    sigma = np.random.rand(k)
    gama = np.zeros((k,n))
    old_miu = np.zeros(k)
    old_sigma = np.zeros(k)
    old_weight = np.zeros(k)
    iter = 0
    while True:
        cluster = [[] for i in range(k)]
        for i in range(k):
            old_miu[i] = miu[i]
            old_sigma[i] = sigma[i]
            old_weight[i] = weight[i]
        post = np.zeros(k)
        for i in range(k):
            for j in range(n):
                for m in range(k):
                    pdf = (1.0/np.sqrt(2*np.pi*sigma[m]))* np.exp(-(data[j]-miu[m])**2/(2*sigma[m]))
                    if pdf == 0:
                        pdf += 0.0001
                    post[m] = weight[m] * pdf
                gama[i][j] = post[i]/float(np.sum(post))
        for j in range(n):  
            cluster[np.argmax(gama[:,j])].append(data[j])
        # updata miu
        for i in range(k):
#            print("np.sum(gama[i]:",np.sum(gama[i]))
            miu[i] = np.sum(gama[i]*data)/np.sum(gama[i])
        # updata sigma
        for i in range(k):
            sigma[i] = np.sqrt(np.sum(gama[i]*(data - miu[i])**2)/np.sum(gama[i]))
        # updata 
        for i in range(k):
            weight[i] = np.sum(gama[i])/n
        if np.sum(np.abs(old_miu - miu)) < threshold and np.sum(np.abs(old_weight - weight)) < threshold and np.sum(np.abs(old_sigma - sigma)):
            break
        if iter > 300:
            break
        iter += 1
#        print("iter:",iter)
#    return gama
    return weight,miu,sigma,cluster


if __name__ == '__main__':
    
    n = len(sys.argv)
    if n == 2:
        data = np.loadtxt(sys.argv[1], delimiter=',')

    k =2
    weight,miu,sigma,cluster = em(data,k)
    print("weight:",weight)
    print("miu:",miu)
    print("sigma:",sigma)
    for i in range(k):
        output = 'em' + str(i) + '.csv'
        np.savetxt(output, cluster[i], delimiter=',')


