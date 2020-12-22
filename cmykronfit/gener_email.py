

# random del 15 个nodes  email.txt
import torch
import pickle
import random
import numpy as np
import networkx as nx 
from networkx.convert import from_dict_of_dicts
from networkx.classes.graph import Graph

seed = 1978
np.random.seed(seed)
random.seed(seed)
adj_address ="/data/chenmy/voter/seed1051email14315000-adjmat.pickle"
with open(adj_address,'rb') as f:
    adj = pickle.load(f,encoding='latin1')
print("adj sz",adj.shape)
def random_del_graph(adj,seed):
    G = nx.from_numpy_matrix(adj) 
    np.random.seed(seed)
    node_order = list(G.nodes)
    node_order_r = np.random.permutation(node_order)
    new_order_graph = dict()
    for node in node_order_r:
        new_order_graph.update({node: G[node]})
    
    new_order_graph = from_dict_of_dicts(new_order_graph, create_using = Graph)
    new_object_matrix = nx.adjacency_matrix(new_order_graph).todense()
    return node_order,node_order_r,new_object_matrix 

node_order,node_order_r,new_object_matrix = random_del_graph(adj,seed)
new_adj = new_object_matrix[:-15,:-15]
print(new_adj.shape)
address = "/data/chenmy/voter/seed1051email128-adjmat.pickle"
with open(address,'wb') as f:
    pickle.dump(new_adj,f)
    print("done")