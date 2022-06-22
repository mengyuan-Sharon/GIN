# used for networkx version 1.11 and python 2.7
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import math
import pickle
from socialnx import *

# 一些函数
# generate the network
def generate_network(g_type='random', n_node=5, average_degree=3):
    # generate ba scale free network
    if g_type == 'ba':
        BA = nx.random_graphs.barabasi_albert_graph(N_Node, 5) # 连边增长 从 2->5
        print('生成的是BA网')
        print(len(BA.edges()))
        return BA
    elif g_type == 'ws':
        WS = nx.random_graphs.watts_strogatz_graph(N_Node, Ws_Nei, Ws_P)
        print('生成的是ws网')
        print(len(WS.edges()))
        return WS
    elif g_type == 'renyi':
        renyi = nx.random_graphs.erdos_renyi_graph(N_Node, renyi_p)
        print('生成的是er网')
        print(len(renyi.edges()))
        return renyi
    elif g_type == 'email':
        path = './email.txt'
        dg = nx.Graph()        
        for i in range(n_node):
            dg.add_node(i, value=random.randint(0, 1))
        with open(path,"r") as f:    
            edges = [(int(i.split(" ")[0])-1,int(i.split(" ")[1])-1) for i in f.read().split("\n")]
            print(edges)
            dg.add_edges_from(edges)
        return dg
    elif g_type == 'football':
        fb = football()
        return fb
# get the innode of each node
# return:{0:[1,2,3],1:[0,4]...}
def get_innode(adj):
    innodes = {}
    for i in range(adj.shape[0]):
        innode = []
        for j in range(adj.shape[0]):
            if adj[j][i] == 1:
                innode.append(j)
        innodes[i] = innode
    return innodes

# init node data randomly
def init_node(dg):
    if G_Type == "football":
        for i in dg.nodes():
            dg.nodes[i]['value'] = random.randint(0, 1)
    else:
        for i in range(dg.number_of_nodes()):
            dg.nodes[i]['value'] = random.randint(0, 1)

# let the net spread by probility
def spread_prob(dg, DYN, step=100):
    node_num = dg.number_of_nodes()
    # data to be returned
    data = []
    # add initial value to data
    origin_val = []
    for i in range(node_num):
        origin_val.append(dg.nodes[i]['value'])
    data.append(origin_val)
    # control the circulates
    run = 0
    # step is the only limitation because there is no conception like attractor and so on...
    while run < step:
        run += 1
        # each step
        next_val = []
        # if DYN is voter
        if DYN == 'voter':
            for i in range(node_num):
                # num for neighbors who vote for agree
                k = 0.
                # num for all neighbors
                m = len(innodes[i])
                for iter, val in enumerate(innodes[i]):
                    if dg.nodes[val]['value'] == 1:
                        k += 1.
                try:
                    if random.random() < k / m:
                        next_val.append(1)
                    else:
                        next_val.append(0)
                except ZeroDivisionError:
                    if random.random() < 0.5:
                        next_val.append(1)
                    else:
                        next_val.append(0)


        # print(next_val)
        # set value to the net
        if G_Type == "football":
            for i in dg.nodes():
                dg.nodes[i]['value'] = next_val[i]
        else:
            for i in range(node_num):
                dg.nodes[i]['value'] = next_val[i]
        # just add to data to record
        data.append(next_val)
    return np.array(data)

#  global config
# set seed
seed = 2051
np.random.seed(seed)

# random / all : randomly init for each sample / init all 2^N_Node states
Init_Way = 'random'
# Reinit_Time = 1024
# Config of network topology
N_Node = 115


# email 
# N_Node = 143
# G_Type = 'email'
# Goal_Data_Num = 30000

# BA 
# G_Type = 'ba'
# Goal_Data_Num = 20000

# ws 
# G_Type = 'ws'
# Goal_Data_Num = 60000
# Ws_Nei = 4
# Ws_P = 0.3

# Average_Degree = 4
# renyi_p = 0.04

# football

G_Type = 'football'
Goal_Data_Num = 20000

# Config of Dyn:table/prob
DYN_Type = 'prob'
DYN = 'voter'
# config of way to detect inherent character of this network
Draw_Grow_Step = 50
# Draw_Hm_Time = 20

# mark
mark = seed
# store folder
Store_Folder = '/data/chenmy/voter/'


dg = generate_network(g_type=G_Type)
print(len(dg))
adj = nx.adjacency_matrix(dg).toarray()
print("edges",adj.sum(0))
# innode of each node

innodes = get_innode(adj)

print("analyze of the graph")

# generate data
all_data = np.array([[-1]])
# has been explored
has_explored = []
datas = []

# in this way we aim to generate data until final number of all data reachs goal data num\
i = 0
while len(has_explored) < Goal_Data_Num:
    print('how many has we explored')
    # if i % 10 == 0:
    print(len(has_explored))

    # print('initial time-----'+str(i))
    # init node with a perticular num,if net is too big then use random init instead
    if Init_Way == 'random':
        init_node(dg)
    # init state
    init_state = []
    if G_Type == "football":
        for j in dg.nodes():
            init_state.append(dg.nodes[j]['value'])   
    else:
        for j in range(N_Node):
            init_state.append(dg.nodes[j]['value'])
    # if this state has been explored
    if init_state in has_explored:
        continue
    else:
        has_explored.append(init_state)

    # spread
    if DYN_Type == 'prob':
        data = spread_prob(dg, DYN, step=Draw_Grow_Step)
        datas.append(data)

    # make each [a,b,c,d] to [a,b,b,c,c,d]
    # (2,3)means(2step,3node)

    # if only one point ,that means it is a eden state and a fix point
    if data.shape[0] == 1:
        temp = np.zeros((2, data.shape[1]))
        temp[0] = data
        temp[1] = data
        data = temp
    expand_data = np.zeros((2 * data.shape[0] - 2, data.shape[1]))
    for j in range(data.shape[0]):
        # add to has explored
        if j < data.shape[0] - 1:
            cur_state = data[j].tolist()
            # if dyn type is table, then we have to make sure weather the state is explored or not
            if DYN_Type == 'table':
                if cur_state not in has_explored:
                    has_explored.append(cur_state)
            # dyn type is prob means we just have to record how many state we have visited
            elif DYN_Type == 'prob':
                has_explored.append(cur_state)

        # generate data to use
        if j == 0:
            expand_data[0] = data[0]
        elif j == data.shape[0] - 1:
            expand_data[expand_data.shape[0] - 1] = data[j]
        else:
            # j between first and last
            expand_data[2 * j - 1] = data[j]
            expand_data[2 * j] = data[j]
    # print(expand_data)
    # concat data in every step
    if all_data[0][0] == -1:
        all_data = expand_data
    else:
        all_data = np.concatenate((all_data, expand_data), axis=0)

print(all_data)
print(all_data.shape)
# change the shape from(step,node_num) => (step,node_num,1)
all_data = all_data[:, :, np.newaxis]
print(all_data.shape)

# save the data
# save time series data
serise_address = Store_Folder + "seed"+str(mark) + str(G_Type) + str(N_Node) + str(Goal_Data_Num) + '-series.pickle'
with open(serise_address, 'wb') as f:
    pickle.dump(all_data, f)

# save adj mat
adj_address = Store_Folder + "seed"+str(mark) + str(G_Type) + str(N_Node) + str(Goal_Data_Num) + '-adjmat.pickle'
with open(adj_address, 'wb') as f:
    pickle.dump(adj, f)

print("path",adj_address,serise_address)