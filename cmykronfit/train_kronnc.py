# random del 15 个nodes  email.txt
import torch
import pickle
import random
import numpy as np
import networkx as nx 
from networkx.convert import from_dict_of_dicts
from networkx.classes.graph import Graph
from kronEM import *
seed =1900
np.random.seed(seed)
random.seed(seed)

adj_address ="/data/chenmy/voter/seed1051email128-adjmat.pickle"
with open(adj_address,'rb') as f:
    objective_adj = pickle.load(f,encoding='latin1')
objective_adj = np.array(objective_adj)
# print(objective_adj)
Ground_truth_adj = torch.FloatTensor(objective_adj)
sz = Ground_truth_adj.shape[0]
k = int(np.log2(sz))
remove_proportion = 0.01
del_num = int(sz*remove_proportion)
mask_un_obs,mask_obs,miss_idx = missing_label(sz,remove_proportion)
G = Ground_truth_adj*mask_obs
nG = G.data.numpy()
H = G.clone()
# print(2,(abs(H.data.numpy()-objective_adj)*mask_obs.data.numpy()).sum())
init_z = Ground_truth_adj*mask_un_obs
missing_edges = int(init_z.sum())
print("miss nodes ",miss_idx)
# initial partial z with fixed missing edges num
z_ele_num = int(mask_un_obs.sum())
z_element_choice = torch.randperm(z_ele_num)[:missing_edges]
z_element = torch.nonzero(mask_un_obs)
init_z_edges = torch.index_select(z_element,0,z_element_choice)
H[init_z_edges[:,0],init_z_edges[:,1]] = 1
# print(abs((H - Ground_truth_adj)*mask_obs).sum())
# print(G,H,z_element_choice)
# initial kronecker
p0 = torch.FloatTensor([[0.9,0.7],[0.7, 0.3]])#torch.FloatTensor([[0.4408, 0.1770],[0.4951, 0.2585]])  # 
generator = kronecker_Generator(p0,k,2)
Pk = generator.generator_adjacency()
init_H = H.data.numpy()
label_non_obs = mask_un_obs.data.numpy()
init_Pk = Pk.detach().data.numpy()
perm = np.arange(sz)
print(abs((init_H- objective_adj)*mask_obs.data.numpy()).sum())

warmup=1
N = 1 
iterstep =1
epoch = 3
emlosses2,perm2 = kronEM(iterstep,N,warmup,init_H,init_Pk,label_non_obs,epoch,p0,perm,k,objective_adj,label_non_obs)


