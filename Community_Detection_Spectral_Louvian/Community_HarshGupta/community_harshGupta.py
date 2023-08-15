import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import warnings
from networkx.classes.function import *
warnings.filterwarnings("ignore")


adj_mat  = np.array([[0, 1, 0, 0],
                     [1, 0, 0, 1],
                     [0, 0, 0, 1],
                     [0, 1, 1, 0]])


def removeEdges(cluster1, cluster2,G):
    for node in cluster1:
        for adjNode in list(G.adj[node]):
            if adjNode in cluster2:
                G.remove_edge(node,adjNode)
    return G


def get_list_clusters(graph_partition):
    d = {}

    for node,community in graph_partition:
        p = int(community)
        if  p in  d.keys():
            d[p].append(node)
        else:
            d[p] = [node]
    list_clusters = list(d.values())
    return list_clusters,d


def plot(graph_partition, G1,dataset):
    def color_list(n):
        colors= []
        for  i in range(n):
            color = "#" + "%06x" % random.randint(0, 0xFFFFFF)
            colors.append(color)
        return colors
    fig = plt.figure(figsize=(50,50))
    
    list_clusters,_ = get_list_clusters(graph_partition)
    n = len(list_clusters)
    colors = color_list(n)
    # print(dataset)
   
    for i in range(n):
        for j in range(i+1,n):
            removeEdges(list_clusters[i],list_clusters[j],G1)
    pos =nx.spring_layout(G1)
    nx.draw_networkx_edges(G1,pos)
    for i in range(n):
        nx.draw_networkx_nodes(G1,pos,nodelist=list_clusters[i],node_color=colors[i], node_size=50,)   
    
    plt.savefig("Graph_partition_for_{}_{}_clusters.png".format(dataset,n+1))
    


              

############################ Start of Louvian Algorithm Helpers Function ############################

def modularity_matrix(G):
    adj_mat = nx.adjacency_matrix(G).toarray()
    d_i = np.sum(adj_mat,axis=1)
    m  = 1/d_i.sum()       # m = 1/ 2*No.of edges
    d_i = np.reshape(d_i,(len(d_i),1))
    K= m* np.matmul(d_i,d_i.T)    #all (d_i *d_j)/2*(no of edges)
    return m * (adj_mat - K )    # k = di*dj/(2*no of edges)


def modularity_community(M,X):
    modularity_value = 0
    for i in X:
        for j in X:
            modularity_value +=M[i][j]
    return  modularity_value


def initialize_communities_and_partitions(G):
    adj_mat = nx.adjacency_matrix(G).toarray()
    M = modularity_matrix(G)
    nodes = list(range(len(adj_mat)))
    communities = {i:{i} for i in range(len(adj_mat))}
    commMap = [i for i in range(len(adj_mat))]
    return adj_mat, M, nodes,commMap, communities

def delM(M, i,j):
    modularity_j  = modularity_community(M,j) 
    # modularity_j_before  = modularity_j + M[i][i]
    # modularity_j_after =  modularity_j + M[i][i] + sum([ M[i][item] + M[item][i] for item in j] ) 
    return sum([ M[i][item] + M[item][i] for item in j] )

def max_modularity_change(M,nodes,communities,commMap):
    c=[]
    n= len(M)
    mx = 0
    for i in nodes:
        c = []
        mx = 0
        for j in communities:
            difference_Q = delM(M,i,communities[j])                
            if mx< difference_Q:
                mx = difference_Q
                c = j

        if mx>0 and commMap[i]!=c:
            
            communities[commMap[i]].remove(i)
            communities[c].add(i)  ## add i to j community
            
            if len(communities[commMap[i]])==0:
                communities.pop(commMap[i])
            commMap[i] = c
#
                

    return communities, commMap





########################## End of Louvian Algorithm Helpers Function ######################################           





################ Main Functions #################

def import_facebook_data(path):
    return np.genfromtxt(path)


def createSortedAdjMat(partition, nodes_connectivity_list,dataset='facebook'):
    if len(nodes_connectivity_list)==21489: dataset='btc'
    print(dataset)
    n_nodes = int(nodes_connectivity_list.max())
    G = nx.from_edgelist(nodes_connectivity_list)
    
    adj_mat = nx.adjacency_matrix(G).toarray()
    sorted_adj_mat = np.zeros((n_nodes,n_nodes))
    sorted_partition = list(sorted(partition, key = lambda x: x[1])) 

    for i,node1 in zip(sorted_partition,range(n_nodes)):
        for j,node2 in zip(sorted_partition,range(n_nodes)):
            try:
                sorted_adj_mat[node1][node2] = adj_mat[int(i[0])][int(j[0])]
            except:
                pass
    plt.figure(figsize=(10,10))
    plt.imshow(sorted_adj_mat, cmap = "binary_r")
    plt.savefig('SortedAdj_{}_mat.png'.format(dataset))
    return sorted_adj_mat

    
    

   



def import_bitcoin_data(path):
    btc = np.genfromtxt(path,delimiter=',')
    node_list =[]
    for item in btc:
        node_list.append((item[0], item[1],{'weight':item[2]}))
    G = nx.from_edgelist(node_list)
    for component in list(nx. connected_components(G)):
        if len(component)<3:
            for node in component:
                G. remove_node(node)
    # adj_mat = nx.adj_matrix(G).toarray()
    nodes_connectivity_list = np.array(list(G.edges()))

    return nodes_connectivity_list

############################ Spectral Clustering ############################

def spectralDecomp_OneIter(nodes_connectivity_list, plot_=True,dataset="facebook"):
    
    G = nx.from_edgelist(nodes_connectivity_list)
    adj_mat = nx.adjacency_matrix(G).toarray()
    fiedler_vector  = nx.fiedler_vector(G,normalized=True)
    
    nodes = list(nx.nodes(G))
    p1 = np.inf
    p2 = np.inf
    for i in range(len(fiedler_vector)):
        if fiedler_vector[i]<=0 :
            p1 = min(p1,nodes[i])
        else:
            p2 = min(p2,int(nodes[i]))
    
    graph_partition = [[nodes[i],int(p1)] if fiedler_vector[i]<=0 else [nodes[i],int(p2)] for i in range(len(fiedler_vector))]
            
    if plot_: 
        if len(nodes_connectivity_list)==21489: dataset='btc'
        print(dataset)
        plot(graph_partition,G,dataset)
        print("Graph for Spectral CLustering for 2 partitions is Generated")
        fig = plt.figure(figsize=(30,30))
        plt.scatter(range(len(nodes)),sorted(fiedler_vector))
        plt.savefig("SortedFiedler_{}.png".format(dataset))
        print("Sorted fiedler vector generated Generated for 2 partitions")

    # if nd:
    #     return fiedler_vector, adj_mat, graph_partition, nodes
    # else:
    return fiedler_vector, adj_mat, graph_partition

def spectralDecomposition(nodes_connectivity_list, k=8,dataset="facebook"):
    if len(nodes_connectivity_list)==21489: dataset='btc'
    print(dataset)
    i=0
    if dataset=="btc" and k==8: k=5
    
    d_f = {}
    G = nx.from_edgelist(nodes_connectivity_list)
    while(i<k-1):
        fiedler_vector, adj_mat, graph_partition = spectralDecomp_OneIter(nodes_connectivity_list,plot_=False)
        i+=1
        list_cluster,d = get_list_clusters(graph_partition)
        print("Cluster 1 size:{} and cluster2 size: {}".format(len(list_cluster[0]),len(list_cluster[1])))
        removeEdges(list_cluster[0],list_cluster[1],G)
        d_f.update(d)
        n = len(graph_partition)
        
        largest_cluster_size =0
        key = 0
        for item in d_f:
            if largest_cluster_size < len(d_f[item]):
                largest_cluster_size = len(d_f[item])
                key =item
        list_of_nodes = d_f.pop(key)
        
        nodes_connectivity_list = np.array(list(nx.edges(G,list_of_nodes)))
        
        print()
        print('--------No of cluster is ={}--------------'.format(len(d_f)))
    graph_partition = []
    
    for item in d_f:
        for node in d_f[item]:
            graph_partition.append([node,item])
    plot(graph_partition,G,dataset)
    return graph_partition



############################ Louvian ALgorithm ############################


def louvain_one_iter(nodes_connectivity_list,dataset="facebook"):
    if len(nodes_connectivity_list)==21489 : dataset='btc'
    # print(dataset,len(nodes_connectivity_list))
    print("##################### Louvian ALgorithm Started ##################### ")
    G = nx.from_edgelist(nodes_connectivity_list)
    _, M, nodes,commMap, communities = initialize_communities_and_partitions(G)
    itr=0
    initial_comm = len(nodes)
    n_itr=1
    while(itr<n_itr):
        communities,commMap  = max_modularity_change(M,nodes,communities,commMap)
        print("After Number of Iteration : {} , Number of Communities reduced to {}".format(itr+1,len(communities)))
        if initial_comm == len(communities): 
            break
        initial_comm = len(communities)
        itr+=1
        # if itr==n_itr: 
        #     print("Algo Doesn't converge within given itr: {}".format(itr))

    graph_partition = []
 
    for item in communities:
        mn = min(communities[item])
        for i in communities[item]:
            graph_partition.append([i,mn])
    
    # if not is_weighted(G):
    #     plot(graph_partition,G,dataset)
    # print("Communities: ", communities)
    print("##################### Louvian ALgorithm Finished for 1 iteration  ##################### ")

    return graph_partition

if __name__ == "__main__":

    ############ Answer qn 1-4 for facebook data #################################################
    # Import facebook_combined.txt
    # nodes_connectivity_list is a nx2 numpy array, where every row 
    # is a edge connecting i<->j (entry in the first column is node i, 
    # entry in the second column is node j)
    # Each row represents a unique edge. Hence, any repetitions in data must be cleaned away.
    nodes_connectivity_list_fb = import_facebook_data("../data/facebook_combined.txt")
    
    # This is for question no. 1
    # fielder_vec    : n-length numpy array. (n being number of nodes in the network)
    # adj_mat        : nxn adjacency matrix of the graph
    # graph_partition: graph_partitition is a nx2 numpy array where the first column consists of all
    #                  nodes in the network and the second column lists their community id (starting from 0)
    #                  Follow the convention that the community id is equal to the lowest nodeID in that community.
 
    print("################ Q1 Spectral Clustering One Iteration  Started ##################")
    
    fielder_vec_fb, adj_mat_fb, graph_partition_fb= spectralDecomp_OneIter(nodes_connectivity_list_fb)
    
    print("################ Q1 Spectral Clustering One Iteration  Ended ################## \n")
    # This is for question no. 2. Use the function 
    # written for question no.1 iteratetively within this function.
    # graph_partition is a nx2 numpy array, as before. It now contains all the community id's that you have
    # identified as part of question 2. The naming convention for the community id is as before.
    print("################ Q2 Spectral Clustering  Started ##################")

    graph_partition_fb = spectralDecomposition(nodes_connectivity_list_fb)  #k = no of clusters is optional in this.
    
    print("################ Q2 Spectral Clustering Ended ################## \n")
    # # This is for question no. 3
    # # Create the sorted adjacency matrix of the entire graph. You will need the identified communities from
    # # question 3 (in the form of the nx2 numpy array graph_partition) and the nodes_connectivity_list. The
    # # adjacency matrix is to be sorted in an increasing order of communitites.
    print("################ Q3 Created Sorted Adjacency Matrix  Started ##################")
    
    clustered_adj_mat_fb = createSortedAdjMat(graph_partition_fb, nodes_connectivity_list_fb)
    
    print("################ Q3 Adjacency Matrix Creation  Ended ################## \n")


    # # # This is for question no. 4
    # # # run one iteration of louvain algorithm and return the resulting graph_partition. The description of
    # # # graph_partition vector is as before.
    print("################ Q4 Louvian one_itr started ##################")

    graph_partition_louvain_fb = louvain_one_iter(nodes_connectivity_list_fb)

    print("################ Q4 Louvian one_itr started ##################\n \n")

    
    print("################ Dataset 2 Started ##################\n \n")

    # # ############ Answer qn 1-4 for bitcoin data #################################################
    # # # Import soc-sign-bitcoinotc.csv
    nodes_connectivity_list_btc = import_bitcoin_data("../data/soc-sign-bitcoinotc.csv")

    # # # Question 1
    print("################ Q1 Spectral Clustering One Iteration  Started ##################")

    fielder_vec_btc, adj_mat_btc, graph_partition_btc= spectralDecomp_OneIter(nodes_connectivity_list_btc)

    print("################ Q1 Spectral Clustering One Iteration  Ended ################## \n")

    # # Question 2
    print("################ Q2 Spectral Clustering  Started ##################")

    graph_partition_btc = spectralDecomposition(nodes_connectivity_list_btc)



    print("################ Q2 Spectral Clustering  Ended ##################\n")


    # # Question 3
    print("################ Q3 Created Sorted Adjacency Matrix  Started ##################")

    clustered_adj_mat_btc = createSortedAdjMat(graph_partition_btc, nodes_connectivity_list_btc)

    print("################ Q3 Adjacency Matrix Creation  Ended ################## \n")


    # # Question 4

    print("################ Q4 Louvian one_itr started ##################")

    graph_partition_louvain_btc = louvain_one_iter(nodes_connectivity_list_btc)

    print("################ Q4 Louvian one_itr started ##################")
