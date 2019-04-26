# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 13:49:47 2019

@author: 37112
"""

import numpy as np
import matplotlib.pyplot as plt
n1 = 5000
n2 = 3000
μ1 = 5
μ2 = 100
sigma1 = 1
sigma2 = 10
#np.random.seed(0)
s1 = np.random.normal(μ1, sigma1, n1)
s1 = s1.astype(int)
s2 = np.random.normal(μ2, sigma2, n2)
s2 = s2.astype(int)
data = np.hstack((s1,s2))
np.random.shuffle(data)
#print(data)
#plt.hist(data, 30, normed=True)
#plt.show()
np.savetxt('data.csv', data,delimiter=',')