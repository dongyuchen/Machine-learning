# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 18:16:50 2019

@author: 37112
"""

import argparse
import os.path as osp
import numpy as np

def txt2list(filepath):
    data = []
    file = open(filepath)
    lines = file.readlines()
    for line in lines:
        line = line.strip()
# =============================================================================
#         for i in range(len(line)):
#             data.append(line[i])
# =============================================================================
        data.append(line)
    return data

def getProb(string,param):
    proba = 0
    if string == "AA":
        proba = param[0][0]
    elif string == "AC":
        proba = param[0][1]
    elif string == "AG":
        proba = param[0][2]
    elif string == "AT":
        proba = param[0][3]
    elif string == "CA":
        proba = param[1][0]
    elif string == "CC":
        proba = param[1][1]
    elif string == "CG":
        proba = param[1][2]
    elif string == "CT":
        proba = param[1][3]
    elif string == "GA":
        proba = param[2][0]
    elif string == "GC":
        proba = param[2][1]
    elif string == "GG":
        proba = param[2][2]
    elif string == "GT":
        proba = param[2][3]  
    elif string == "TA":
        proba = param[3][0]
    elif string == "TC":
        proba = param[3][1]
    elif string == "TG":
        proba = param[3][2]
    elif string == "TT":
        proba = param[3][3]  
    
    return proba

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Problem-MarkovModel')
#    parser.add_argument('filename', type=str, dest='data',
#                        help='the filename whicn needs to be test', default='test1.txt')
#    parser.add_argument('--inside', type=str, dest='param_in', 
#                        help='the Markov models’ parameters of inside', default='inside.txt')
#    parser.add_argument('--outside', type=str, dest='param_out',  
#                        help='the Markov models’ parameters of outside', default='outside.txt')
    parser.add_argument('filename', type=str, nargs=3,
                        help='the filename whicn needs to be test')
    args = parser.parse_args()

    return args

if __name__ == '__main__':

    args = parse_args()#model setting
    
    in_path = osp.abspath(osp.dirname(__file__))#root dir path
#    in_file = osp.join(in_path , args.data)# input file path
#    param_in_name = osp.join(in_path , args.param_in)# inside table paramater path
#    param_out_name = osp.join(in_path , args.param_out)# outside table paramater path
    in_file = osp.join(in_path , args.filename[0])# input file path
    param_in_name = osp.join(in_path , args.filename[1])# inside table paramater path
    param_out_name = osp.join(in_path , args.filename[2])

    iTab = np.loadtxt(param_in_name,)
    nTab = np.loadtxt(param_out_name)
    lrTab = np.log2(iTab) - np.log2(nTab)
    data = txt2list(in_file)
#    print(lrTab)
    for i in range(len(data)):
        score = 0
        for j in range(len(data[i])-1):
            prob = getProb(data[i][j]+data[i][j+1],lrTab) 
            score += prob
        if score > 0:
            res = "inside"
        else:
            res = "outside"
        
        print(score,res)    
