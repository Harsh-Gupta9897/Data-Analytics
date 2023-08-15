from ctypes import sizeof
from distutils import dir_util
from locale import normalize
import re
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.sparse.csgraph import laplacian
import scipy
import networkx as nx
import random

adj_mat  = np.array([[0, 1, 0, 0],
                     [1, 0, 0, 1],
                     [0, 0, 0, 1],
                     [0, 1, 1, 0]])



def spectral_decomposition(adj_mat):
    '''
    Input:     adj_mat : list of list
               L : normalized laplcian
    output:    eig_Val ,fiedler_vector and sorted fiedler vector
    '''
    L = laplacian(adj_mat,normed=True)
    
    eig_val, eig_vector = np.linalg.eigh(L)
    eig_vector_sorted = sorted(eig_vector.T[1])
    
    return eig_val[1],eig_vector.T[1], eig_vector_sorted



def modularity_matrix(G):
    '''
        It return Modularity Matrix of Given Graph
    '''
    adj_mat = nx.adjacency_matrix(G).toarray()
    d_i = np.sum(adj_mat,axis=1)
    m  = 1/d_i.sum()       # m = 1/ 2*No.of edges
    d_i = np.reshape(d_i,(len(d_i),1))
    K= m* np.matmul(d_i,d_i.T)    #all (d_i *d_j)/2*(no of edges)
    return m * (adj_mat - K )    # k = di*dj/(2*no of edges)


def modularity_community(M,X):
    ''' Input : M is Modularity Matrix of Graph
                X : is list of nodes of given community
        Output : modularity_value of given community
    '''
    modularity_value = 0
    for i in X:
        for j in X:
            modularity_value +=M[i][j]
    return  modularity_value


def initialize_communities_and_partitions(G):
    '''
        Input: Graph G
        Output:  Modularity Matrix
                 nodes : list of nodes
                Communities : dict of set of communities map to node 

    '''
    adj_mat = nx.adjacency_matrix(G).toarray()
    M = modularity_matrix(G)
    nodes = list(range(len(adj_mat)))
    communities = {i:{i} for i in range(len(adj_mat))}
    commMap = [i for i in range(len(adj_mat))]
    return adj_mat, M, nodes,commMap, communities

def delQ(M, i,j):
    modularity_j  = modularity_community(M,j) 
    # modularity_j_before  = modularity_j + M[i][i]
    # modularity_j_after =  modularity_j + M[i][i] + sum([ M[i][item] + M[item][i] for item in j] ) 
    return sum([ M[i][item] + M[item][i] for item in j] )

def max_modularity_change(M,nodes, communitites):
    c=[]
    n= len(M)
    mx = 0
    i_final =0
    for i in nodes:
        for j in communitites:
            difference_Q = delQ(M,i,j)
            if mx< difference_Q:
                mx = difference_Q
                c = j
                i = i_final
    # if(mx==0
    return i, c

def louvian_method(G):

    adj_mat, M, nodes,commMap, communities = initialize_communities_and_partitions(G)
    i,j  = max_modularity_change(M,nodes,communities)
    print(i,j)


    
def color_list(n):
    colors= []
    for  i in range(n):
        color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
        colors.append(color)
    return colors


    




# def louvian_algorithm(adj_mat):

# p= np.sum(adj_mat,axis=1)

# p = np.reshape(p,(len(p),1))
# print(np.matmul(p,p.T))