# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 18:27:10 2019

@author: 37112
"""
import numpy as np
import argparse
import os.path as osp

def txt2list(filepath):
    data = []
    file = open(filepath)
    lines = file.readlines()
    for line in lines[1:]:
        line = line.strip()
        for i in range(len(line)):
            data.append(line[i])
            
    return data

def dataTrans(data,n,symbols):
    res = np.zeros_like(data)
    for i in range(len(data)):
        for j in range(n):
            if data[i].upper() == symbols[j].upper():
                res[i] = j

    return res

def hmm2list(filepath):
    param = []
    file = open(filepath)
    lines = file.readlines()
    for line in lines:
        line = line.strip()
        param.append(line.split())
    return param

def viterbi(ob, state, trans, emis, pi):
    # max_p: record the max_probability at every time for every state，（i,j）i:time j:max_p of hide state     
    max_p = np.zeros((len(ob),len(state)),dtype = np.float)
    # path: record path at time i the best path for i-1
    path = np.zeros((len(ob),len(state)),dtype = np.uint8)
    # initialize(pi state)
    for i in range(len(state)):
        # max_p[0][i] means the max prob at init time for state(i)
        # prob = pi[i] * emis[state[i]][ob[0]]
        max_p[0][i] = np.log2(float(pi[i])) + np.log2(float(emis[state[i]][int(ob[0])]))
        path[0][i] = i
    # iteration(time from 2 to t)
    # at this time max_p has recorded one hidden state prob
    for i in range(1, len(ob)):  
        max_item = np.zeros(len(state))
        # compute every hidden state prob for current
        for j in range(len(state)):  
            item = np.zeros(len(state))
            for k in range(len(state)):  # compute different state prob
                score = max_p[i - 1][k] + np.log2(float(emis[state[j]][int(ob[i])])) + np.log2(float(trans[state[k]][state[j]]))
                item[state[k]] = score
            max_item[state[j]] = max(item)
            # When the probability of the J state is the highest at the current moment, its precursor node is recorded
            # np.argwhere(item == max(item)) find item's max index as item record every precursor node prob
            path[i][state[j]] = np.argwhere(item == max(item))
        # Adds the results of a single state to the total list[max_p]
        max_p[i] = max_item
    #lastpath record last path
    lastpath = []
    #judge the last time which state has the max prob
    point = int(np.argwhere(max_p[-1] == max(max_p[-1])))
    lastpath.append(point)
    for i in range(len(ob) - 1, 0, -1):
        lastpath.append(path[i][point])
        point = int(path[i][point])
    lastpath.reverse()
    return lastpath

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Problem-MarkovModel')
#    parser.add_argument('--filename', type=str, dest='data', 
#                        help='the filename whicn needs to be test', default='example.fa')
#    parser.add_argument('--inside', type=str, dest='param', 
#                        help='the Markov models’ parameters of inside', default='example.hmm')
    parser.add_argument('filename', type=str, nargs=2, 
                        help='enter 2 filename, test file and parameter file')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()#model setting
    
    in_path = osp.abspath(osp.dirname(__file__))#root dir path
#    in_file = osp.join(in_path , args.data)# input file path
#    param_in_name = osp.join(in_path , args.param)# inside table paramater path
    in_file = osp.join(in_path , args.filename[0])# input file path
    param_in_name = osp.join(in_path , args.filename[1])# inside table paramater path

    data = txt2list(in_file)
    para = hmm2list(param_in_name)
    state_n = int(para[0][0]) # the number of state
    state = [int(x) for x in range(state_n)]
    hidden_state = [int(x) for x in range(1,state_n + 1)]
    symbols_n = int(para[0][1]) # the number of symbols
    symboles = [x for x in para[0][2]]
    pi = para[1] # initial parameter
    tran = [] # transmision matrix
    emis = [] # emission matrix
    ob = dataTrans(data,symbols_n,symboles) #data transform into number form
    for i in range(state_n):
        tran.append(para[2+i][0:state_n])
        emis.append(para[2+i][state_n:])
    result = viterbi(ob, state, tran, emis, pi)
    start = 1
    a = result[0]
#    outfile ='E:/course/6435 multi-model/homework 3/Problem-HMM/output1.txt'
    if result[0] == 1:
        b = 1
    else:
        b = 0
    for k in range(1,len(result)):
        if result[k] != a:
            end = k
#            if osp.exists(outfile):    
#                f = open(outfile,'a')
#                f.writelines(['\n',str(start)," ", str(end), " ", "state ", str(hidden_state[a])])
#            else:
#                f = open(outfile,'w')
#                f.writelines([str(start)," ", str(end), " ", "state ", str(hidden_state[a])])
            print(start," ", end, " ", "state ", hidden_state[a])
            start = k + 1
            a = result[k]
            b += 1
#            if result[k] == 1:
#                b += 1
    if start != len(result):
        print(start," ", len(result), " ", "state ", hidden_state[int(result[-1])])
    print("There are total {} segments of the genome are in state B".format(b))