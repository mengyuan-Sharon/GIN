# gene 网络
# %load data_generator_cml.py
import networkx as nx
import scipy.sparse
import argparse
import pandas as pd
import torch
import numpy as np
import pickle
import os

def read_txt(data):
    g = nx.read_edgelist(data, create_using=nx.DiGraph())
    return g
def add_edge(adj,g0,g1,value):
	if int(value) == 1:
		adj[int(g0[1:])-1][int(g1[1:])-1] = 1
	return adj
def read_gene():
    node = 100
    adj_path = "/data/chenmy/insilico_size100_1_goldstandard.tsv"
    adj_data = pd.read_csv(adj_path,sep='\t')
    # initialize it to be zero
    adj = torch.zeros(node,node)
    adj = add_edge(adj,adj_data.columns[0],adj_data.columns[1],adj_data.columns[2])
    list_value = list(adj_data.values)
    for i in range(int(len(list(adj_data.values)))):
        adj = add_edge(adj,list_value[i][0],list_value[i][1],list_value[i][2])
    return adj
# G = read_txt("email.txt")
# for i in range(100):
#     seed = np.random.randint(300)+800
# #     print(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     g = nx.random_graphs.erdos_renyi_graph(100,0.04)
#     a = nx.adjacency_matrix(g).todense().sum(0) == 0
# #     print(a)
#     if a.any() :
#         print("wrong")
#     else:
#         print(seed)
use_cuda = torch.cuda.is_available()
cuda_number = 3
# torch.set_num_threads(1)
torch.cuda.set_device(cuda_number)

parser = argparse.ArgumentParser()
parser.add_argument('--nodes', type=int, default=100, help='Number of nodes, default=10')
parser.add_argument('--samples', type=int, default=70, help='Number of samples in simulation, default=7000')
parser.add_argument('--prediction-steps', type=int, default=2, help='prediction steps, default=10')
parser.add_argument('--evolving-steps', type=int, default=100, help='evolving steps, default=100')
parser.add_argument('--lambd', type=float, default=3.5, help='lambda in logistic map, default=3.6')
parser.add_argument('--coupling', type=float, default=0.2, help='coupling coefficent, default=0.2')
parser.add_argument('--data-name', type=str, default= 'cmlWS0.2-', help='data name to save')
parser.add_argument('--seed', type=int, default=2000, help='data random seed')
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def logistic_map(x, lambd=args.lambd):
    # return 1 - lambd * x ** 2
    return lambd * x * (1 - x)

class CMLDynamicSimulator():
    def __init__(self, batch_size, sz, s):
        self.s = s
        self.thetas = torch.rand(batch_size, sz)
        # self.G = nx.random_graphs.erdos_renyi_graph(sz,0.04) # 随机网
        self.G = nx.random_graphs.watts_strogatz_graph(sz,4,0.2) # WS 网
        # self.G = nx.random_graphs.barabasi_albert_graph(sz,2)      

        A = nx.to_scipy_sparse_matrix(self.G, format='csr')
        n, m = A.shape
        diags = A.sum(axis=1)
        D = scipy.sparse.spdiags(diags.flatten(), [0], m, n, format='csr')

        self.obj_matrix = torch.FloatTensor(A.toarray())
        self.inv_degree_matrix = torch.FloatTensor(np.linalg.inv(D.toarray()))

        if use_cuda:
            self.thetas = self.thetas.cuda()
            self.obj_matrix = self.obj_matrix.cuda()
            self.inv_degree_matrix = self.inv_degree_matrix.cuda()

    def SetMatrix(self, matrix):
        self.obj_matrix = matrix
        if use_cuda:
            self.obj_matrix = self.obj_matrix.cuda()

    def SetThetas(self, thetas):
        self.thetas = thetas
        if use_cuda:
            self.thetas = self.thetas.cuda()

    def OneStepDiffusionDynamics(self):
        self.thetas = (1 - self.s) * logistic_map(self.thetas) + self.s *                       torch.matmul(torch.matmul(logistic_map(self.thetas), self.obj_matrix), self.inv_degree_matrix)
        return self.thetas
if __name__ == '__main__':
    # 传入参数
    num_nodes = args.nodes
    num_samples = args.samples
    prediction_steps = args.prediction_steps
    evolve_steps = args.evolving_steps
    lambd = args.lambd
    coupling = args.coupling

    # 生成数据
    simulator = CMLDynamicSimulator(batch_size=num_samples, sz=num_nodes, s=coupling)
    simulates = np.zeros([num_samples, num_nodes, evolve_steps, 1])
    sample_freq = 1
    for t in range((evolve_steps + 1) * sample_freq):
        locs = simulator.OneStepDiffusionDynamics()
        if t % sample_freq == 0:
            locs = locs.cpu().data.numpy() if use_cuda else locs.data.numpy()
            simulates[:, :, t // sample_freq - 1, 0] = locs
    data = torch.Tensor(simulates)
    print('原始数据维度：', simulates.shape)

    # 数据切割
    prediction_num = data.size()[2] // prediction_steps
    for i in range(prediction_num):
        last = min((i + 1) * prediction_steps, data.size()[2])
        feat = data[:, :, i * prediction_steps: last, :]
        if i == 0:
            features = feat
        else:
            features = torch.cat((features, feat), dim=0)
    # features数据格式：sample, nodes, timesteps, dimension=1)
    print('切割后的数据维度：', features.shape)

    # shuffle
    features_perm = features[torch.randperm(features.shape[0])]

    # 划分train, val, test
    train_data = features_perm[: features.shape[0] // 7 * 5, :, :, :]
    val_data = features_perm[features.shape[0] // 7 * 5: features.shape[0] // 7 * 6, :, :, :]
    test_data = features_perm[features.shape[0] // 7 * 6:, :, :, :]

    print(train_data.shape, val_data.shape, test_data.shape)

    results = [simulator.obj_matrix, train_data, val_data, test_data]
    if not os.path.exists('/data/chenmy/cml') :
        os.mkdir("/data/chenmy/cml/")
    data_name = "/data/chenmy/cml/"+str(args.seed)+args.data_name+str(args.nodes)+'-'+str(train_data.shape[0])+".pickle"
    print("data_name",data_name)

    with open(data_name, 'wb') as f:
        pickle.dump(results, f)