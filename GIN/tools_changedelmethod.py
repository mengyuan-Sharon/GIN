import torch
import numpy as np
import os
import math
import pickle
import random
import torch.nn.functional as F
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import networkx as nx
from networkx.convert import from_dict_of_dicts
from networkx.classes.graph import Graph
from sgm import sgraphmatch

from sklearn.metrics import confusion_matrix,f1_score,roc_curve,auc

# if use cuda
#use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")


# ----------------- stop control ------------------------

# record one experiment
def record_stop(start_time,HYP):
    # if already exist
    if os.path.exists('stop_control.pickle'):
        with open('stop_control.pickle','rb') as f:
            stop_cont_obj = pickle.load(f)
        stop_cont_obj[start_time] = {'stop':False,'HYP':HYP}
    # if file dont exist
    else:
        stop_cont_obj = {start_time:{'stop':False,'HYP':HYP}}
    # save the result
    with open('stop_control.pickle','wb') as f:
        pickle.dump(stop_cont_obj,f)
    return True

# whether to stop early
def if_stop(start_time):
    with open('stop_control.pickle','rb') as f:
        stop_control = pickle.load(f)
    if stop_control[start_time]['stop'] == True:
        return True
    else:
        return False

# ----------------- stop control ------------------------

# read data

def states_r(states,order,new_order,data_type = 'cml'):
    new_states = torch.zeros_like(states)
    if data_type== 'cml':
        for i,j in zip(order,new_order):
            new_states[:,i,:,:] = states[:,j,:,:]
    if data_type == 'voter':
        if len(new_states.shape)==2:
            for i,j in zip(order,new_order):
                new_states[:,i] = states[:,j]

        else: 
             for i,j in zip(order,new_order):
                new_states[:,i,:] = states[:,j,:]
    
    return new_states

def random_del_graph(adj,seed,data_type = 'voter'):
    if data_type== 'cml':
        G = nx.from_numpy_matrix(adj.cpu().data.numpy())
    if data_type== 'voter':
        G = nx.from_numpy_matrix(adj)
        
    np.random.seed(seed)
    node_order = list(G.nodes)
    node_order_r = np.random.permutation(node_order)
    new_order_graph = dict()
    for node in node_order_r:
        new_order_graph.update({node: G[node]})
    
    new_order_graph = from_dict_of_dicts(new_order_graph, create_using = Graph)
    new_object_matrix = torch.FloatTensor(nx.adjacency_matrix(new_order_graph).todense()).cuda()
    return node_order,node_order_r,new_object_matrix 

def load_cml_ggn(data_path,batch_size = 128,node=10,seed=2050):
    # data_path = '/data/zzhang/data/cml/data_lambd3.6_coupl0.2_node'+str(node)+'.pickle'
    # data_path = '/home/chenmy/GGN_cml/cmlGGN/newtrain/data/cml/ER_p0.04100300.pickle'
    
    with open(data_path, 'rb') as f:
        object_matrix, train_data, val_data, test_data = pickle.load(f) # (samples, nodes, timesteps, 1)
    
    print('\nMatrix dimension: %s Train data size: %s Val data size: %s Test data size: %s'
          % (object_matrix.shape, train_data.shape, val_data.shape, test_data.shape))    
    
    order,new_order,object_matrix = random_del_graph(object_matrix,seed,'cml')
    train_data = states_r(train_data,order,new_order,'cml')
    val_data = states_r(val_data,order,new_order,'cml')
    test_data = states_r(test_data,order,new_order,'cml')
    
    train_loader = DataLoader(train_data[:], batch_size=batch_size, shuffle=False)#
    val_loader = DataLoader(val_data[:], batch_size=batch_size, shuffle=False) # 记得改回来
    test_loader = DataLoader(test_data[:], batch_size=batch_size, shuffle=False) # 记得改回来
    return train_loader,val_loader,test_loader,object_matrix

def load_bn_ggn(series_address,adj_address,batch_size,seed):
    
    with open(adj_address,'rb') as f:
        edges = pickle.load(f,encoding='latin1')
    # time series data
    with open(series_address,'rb') as f:
        info_train = pickle.load(f,encoding='latin1')
    
    # 调整graph 的顺序
    node_order,node_order_r,new_object_matrix  = random_del_graph(edges,seed,'voter')
    
    # 即将用到的数据，先填充为全0
    data_x = np.zeros((int(info_train.shape[0]/2),info_train.shape[1],2))
    data_y = np.zeros((int(info_train.shape[0]/2),info_train.shape[1]))


    # 预处理成分类任务常用的数据格式
    for i in range(int(info_train.shape[0] / 2)):
        for j in range(info_train.shape[1]):
            if info_train[2*i][j][0] == 0.:
                data_x[i][j] = [1,0]
            else:
                data_x[i][j] = [0,1]
            if info_train[2*i+1][j][0] == 0.:
                data_y[i][j] = 0
            else:
                data_y[i][j] = 1

    # random permutation
    indices = np.random.permutation(data_x.shape[0])
    data_x_temp = torch.DoubleTensor([data_x[i] for i in indices])
    data_y_temp = torch.LongTensor([data_y[i] for i in indices])
   
    # states 顺序
    data_x = states_r(data_x_temp,node_order,node_order_r,'voter')
    data_y = states_r(data_y_temp,node_order,node_order_r,'voter')
    
    # seperate train set,val set and test set
    # train / val / test == 5 / 1 / 1 
    train_len = int(data_x.shape[0] * 5 / 7)
    val_len = int(data_x.shape[0] * 6 / 7)
    
    # seperate
    feat_train = data_x[:train_len]
    target_train = data_y[:train_len]
    feat_val = data_x[train_len:val_len]
    target_val = data_y[train_len:val_len]
    feat_test = data_x[val_len:]
    target_test = data_y[val_len:]

    # put into tensor dataset
    train_data = TensorDataset(feat_train, target_train)
    val_data = TensorDataset(feat_val, target_val)
    test_data = TensorDataset(feat_test,target_test)

    # put into dataloader
    train_data_loader = DataLoader(train_data, batch_size=batch_size,drop_last=False)
    valid_data_loader = DataLoader(val_data, batch_size=batch_size,drop_last=False)
    test_data_loader = DataLoader(test_data, batch_size=batch_size,drop_last=False)
    edges = new_object_matrix.clone()

    return train_data_loader,valid_data_loader,test_data_loader,edges

def load_bn_ggn_ori(series_address,adj_address,batch_size = 128,dyn_type='prob'):

    # address
    #adj_address = '/home/zzhang/network_reconstruction/data/table2_bn10/mark-20865-adjmat.pickle'
    # seires_address = '/home/zzhang/network_reconstruction/data/table2_bn10/mark-20865-series.pickle'

    # 5/7 for training, 1/7 for validation and 1/7 for test



    # adj mat
    with open(adj_address,'rb') as f:
        edges = pickle.load(f,encoding='latin1')
    # time series data
    with open(series_address,'rb') as f:
        info_train = pickle.load(f,encoding='latin1')

    use_state = info_train.shape[0]*0.4
    # print(use_state)
    # if too large...
    if info_train.shape[0] > 100000:
        info_train = info_train[:100000]
    info_train_list = info_train.tolist()
    has_loaded = []
    i = 0
    while len(has_loaded) < use_state:
        # if dyn type == table then we have to make sure that each state we load is different
        if dyn_type == 'table':
            # print(i)
            if info_train_list[i] not in has_loaded:
                has_loaded.append(info_train_list[i])
            i = i+2
        elif dyn_type == 'prob':
            #print("i",i)
            #print("len",len(has_loaded))
            # then we dont require they are different
            has_loaded.append(info_train_list[i])
            i = i+2
        else:
            print('Error in loading')
            debug()
    info_train = info_train[:i+2]

    # 即将用到的数据，先填充为全0
    data_x = np.zeros((int(info_train.shape[0]/2),info_train.shape[1],2))
    data_y = np.zeros((int(info_train.shape[0]/2),info_train.shape[1]))


    # random permutation
    indices = np.random.permutation(data_x.shape[0])
    data_x_temp = [data_x[i] for i in indices]
    data_y_temp = [data_y[i] for i in indices]
    data_x = np.array(data_x_temp)
    data_y = np.array(data_y_temp)


    # 预处理成分类任务常用的数据格式
    for i in range(int(info_train.shape[0] / 2)):
        for j in range(info_train.shape[1]):
            if info_train[2*i][j][0] == 0.:
                data_x[i][j] = [1,0]
            else:
                data_x[i][j] = [0,1]
            if info_train[2*i+1][j][0] == 0.:
                data_y[i][j] = 0
            else:
                data_y[i][j] = 1

    # random permutation
    indices = np.random.permutation(data_x.shape[0])
    data_x_temp = [data_x[i] for i in indices]
    data_y_temp = [data_y[i] for i in indices]
    data_x = np.array(data_x_temp)
    data_y = np.array(data_y_temp)

    # seperate train set,val set and test set
    # train / val / test == 5 / 1 / 1 
    train_len = int(data_x.shape[0] * 5 / 7)
    val_len = int(data_x.shape[0] * 6 / 7)
    # seperate
    feat_train = data_x[:train_len]
    target_train = data_y[:train_len]
    feat_val = data_x[train_len:val_len]
    target_val = data_y[train_len:val_len]
    feat_test = data_x[val_len:]
    target_test = data_y[val_len:]

    # change to torch.tensor
    feat_train = torch.DoubleTensor(feat_train)
    feat_val = torch.DoubleTensor(feat_val)
    feat_test = torch.DoubleTensor(feat_test)
    target_train = torch.LongTensor(target_train)
    target_val = torch.LongTensor(target_val)
    target_test = torch.LongTensor(target_test)

    # put into tensor dataset
    train_data = TensorDataset(feat_train, target_train)
    val_data = TensorDataset(feat_val, target_val)
    test_data = TensorDataset(feat_test,target_test)

    # put into dataloader
    train_data_loader = DataLoader(train_data, batch_size=batch_size,drop_last=False)
    valid_data_loader = DataLoader(val_data, batch_size=batch_size,drop_last=False)
    test_data_loader = DataLoader(test_data, batch_size=batch_size,drop_last=False)
    edges = torch.FloatTensor(edges)

    return train_data_loader,valid_data_loader,test_data_loader,edges

def load_sim_ggn(batch_size=128,node=50,p=25):
    edges_300_10k_p25.pickle
    # with open('./data/zzhang/spring/edges_'+str(node)+'_10k_p'+str(p)+'.pickle','rb') as f:
    with open('edges_300_10k_p25.pickle','rb') as f:
        edges = pickle.load(f)
    with open('edges_300_10k_p25.pickle','rb') as f:
    # with open('./data/zzhang/spring/sim_'+str(node)+'_10k_p'+str(p)+'.pickle','rb') as f:
        loc_np = pickle.load(f)
    with open('./data/zzhang/spring/vel_'+str(node)+'_10k_p'+str(p)+'.pickle','rb') as f:
        vel_np = pickle.load(f)

    # transform to tensor
    # 499*2*5
    loc = torch.from_numpy(loc_np)
    vel = torch.from_numpy(vel_np)
    loc = torch.cat((loc,vel),1)
    P = 2
    sample = int(loc.size(0)/P)
    node = loc.size(2)
    dim = loc.size(1)

    loc = loc.transpose(1,2)# 499,5,4
    data = torch.zeros(sample,node,P,dim)
    for i in range(data.size(0)):
        data[i] = loc[i*P:(i+1)*P].transpose(0,1)

    # cut to train val and test
    train_data = data[:int(sample*5/7)]
    val_data = data[int(sample*5/7):int(sample*6/7)]
    test_data = data[int(sample*6/7):]



    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader,val_loader,test_loader,torch.from_numpy(edges)


def load_kuramoto_ggn(batch_size = 128):
    k_over_kc = 1.1

    train_fp = './data/kuramoto/ERtrain-5000sample-1.1kc10node-100timestep-2vec.npy'
    val_fp = './data/kuramoto/ERval-1000sample-1.1kc10node-100timestep-2vec.npy'
    test_fp = './data/kuramoto/ERtest-1000sample-1.1kc10node-100timestep-2vec.npy'
    adj_fp = './data/kuramoto/ERadj-10sample-1.1kc10node-100timestep-2vec.npy'

    object_matrix = np.load(adj_fp)
    train_data = weighted(np.load(train_fp)[:5000,:,:,:], 0.5/k_over_kc)
    val_data = weighted(np.load(val_fp)[:1000,:,:,:], 0.5/k_over_kc)
    test_data = weighted(np.load(test_fp)[:1000,:,:,:], 0.5/k_over_kc)
    num_nodes = object_matrix.shape[0]
    data_max = train_data.max()

    train_dataset = crop_data(train_data)
    val_dataset = crop_data(val_data)
    test_dataset = crop_data(test_data)

    train_dataset = np.asarray(train_dataset, dtype=np.float32)
    val_dataset = np.asarray(val_dataset, dtype=np.float32)
    test_dataset = np.asarray(test_dataset, dtype=np.float32)

    print('\nMatrix dimension: %s Train data size: %s Val data size: %s Test data size: %s'
          % (object_matrix.shape, train_dataset.shape, val_dataset.shape, test_dataset.shape))
    if use_cuda:
        object_matrix = torch.from_numpy(object_matrix).float().cuda()
        train_dataset = torch.from_numpy(train_dataset).cuda()
        val_dataset = torch.from_numpy(val_dataset).cuda()
        test_dataset = torch.from_numpy(test_dataset).cuda()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader, val_loader, test_loader, object_matrix

#————————————————————————————
# 修改评估函数 计算网络补全的评价指标值

# 计算未知部分的准确率
def match_pre_obj_pre(pre,obj_ori,del_num):
    unpre = pre[:-del_num,-del_num:]
    unobj = obj_ori[:-del_num,-del_num:]
    match = unobj.T@unpre
    index_order = torch.zeros(del_num)
    for i in range(del_num):
        max_match = match[i].argmax()
        match[:,max_match] = -2
        index_order[i]= max_match
        
    index_order = index_order.cuda() if use_cuda else index_order
    print("index_order")
    return index_order

# 计算未知部分的准确率
def match_pre_obj(pre,obj_ori,del_num):
    unpre = pre[:-del_num,-del_num:]
    unobj = obj_ori[:-del_num,-del_num:]
    match = -(unobj.T@torch.log(unpre)+(1-unobj).T@torch.log(1-unpre))
    index_order = torch.zeros(del_num)
    for i in range(del_num):
        max_match = match[i].argmin()
        match[:,max_match] = float("inf")
        index_order[i]= max_match
    index_order = index_order.cuda() #if use_cuda else index_order
    print(index_order)
    return index_order

#def match_p(pre,del_num,index):
   # adj = torch.zeros_like(pre)
   # adj[:-del_num,:-del_num] = pre[:-del_num,:-del_num]
   # adj[:,-del_num:] = torch.index_select(pre[:,-del_num:],1,index.long())
   # adj[-del_num:,:-del_num] = torch.index_select(pre[-del_num:,:-del_num],0,index.long())
   # return adj
    
def match_p(pre,sz,del_num,index):
    p = torch.eye(sz).cuda()
    kn_nodes = sz-del_num
    index = torch.cat((torch.tensor(range(kn_nodes)).cuda(),(index+kn_nodes).long()),0)
    pt = torch.index_select(p,0,index)
    pre_adj = torch.mm(pt,pre).mm(pt.T)
    return pre_adj

def tpr_fpr_part(pre,true_adj,un_mask,kn_un_mask,un_un_mask,del_num):
    '''计算未知部分的准确率'''
    un_metics = cal_tpfp(pre,true_adj,un_mask)
    kn_un_metrics = cal_tpfp(pre,true_adj,kn_un_mask)
    un_un_metrics = cal_tpfp(pre,true_adj,un_un_mask)
    return (un_metics,kn_un_metrics,un_un_metrics)

def cal_tpfp(pre,true_adj,mask):
    true_mask = true_adj[mask.bool()]
    pre_un = pre[mask.bool()]
    err = torch.sum(torch.abs(pre_un - true_mask))
    tp = torch.sum(pre_un[true_mask.bool()])
    fn = torch.sum(1- pre_un[true_mask.bool()])
    tn = torch.sum(1-pre_un[(1-true_mask).bool()])
    fp = torch.sum(pre_un[(1-true_mask).bool()])
    precision = tp/(tp+fp)  
    recall = tp/(tp+fn)
    pf = tn/(tn+fn)
    rf = tn/(tn+fp)
    
    f1 = precision*recall/(precision+recall) + pf*rf/(pf+rf)
    tpr = tp/(tp+fn)
    fpr = fp/(tn+fp)
    return (f1,err,tp,fn,fp,tn,tpr,fpr)

def gumbel_p(generator,node_num,left_mask):
    p = F.softmax(generator.gen_matrix,dim =1)[:,0]
    matrix = torch.zeros(node_num,node_num).cuda()
    un_index = torch.triu(left_mask).nonzero()
    matrix[(un_index[:,0],un_index[:,1])] = p
    out_matrix = matrix + matrix.T
    return out_matrix

def aucs(pre_adj,object_matrix,un_mask,un_un_mask,kn_un_mask):
    roc_auc = cal_auc(pre_adj,object_matrix,un_mask)   
    un_un_roc_auc = cal_auc(pre_adj,object_matrix,un_un_mask)     
    kn_un_roc_auc = cal_auc(pre_adj,object_matrix,kn_un_mask)
    return roc_auc,un_un_roc_auc,kn_un_roc_auc

def cal_auc(pre,true_adj,un_mask):
    # print(1,un_mask)
    pre_un = pre[un_mask.bool()].cpu().detach().numpy()
    true_un = true_adj[un_mask.bool()].cpu().detach().numpy()
    fpr,tpr,threshold = roc_curve(true_un,pre_un)
    roc_auc = auc(fpr,tpr)
    return roc_auc

def part_constructor_evaluator(generator, tests, obj_matrix,sz,del_num):
    precision = []
    
    kn_mask,un_mask,un_un_mask,kn_un_mask = partial_mask(sz,del_num)
    
    pre_adj = gumbel_p(generator,sz,un_mask)
    index_order = match_pre_obj(pre_adj,obj_matrix.cuda(),del_num)
    pre_adj = match_p(pre_adj,sz,del_num,index_order)
    auc_net = aucs(pre_adj,obj_matrix,un_mask,un_un_mask,kn_un_mask)
    
    print("auc:%f,un_un_auc:%f,kn_un_auc:%f"%(auc_net[0],auc_net[1],auc_net[2]))
    
    for t in range(tests):
        obj_matrix = obj_matrix.cuda()
        out_matrix = generator.sample_all(hard = True)#使用hard采样 才能计算tpr 与fpr    
        out_matrix = match_p(out_matrix,sz,del_num,index_order)
        metrics_all = tpr_fpr_part(out_matrix,obj_matrix,un_mask,kn_un_mask,un_un_mask,del_num)
        precision.append(metrics_all)
    
    (f1,err_net,tp,fn,fp,tn,tpr,fpr) = np.mean([precision[i][0] for i in range(tests)],0)
    un_precision = (f1,err_net,tp,fn,fp,tn,tpr,fpr)
    print("f1:%f,err_net:%d,tp:%d,fn:%d,fp:%d,tn:%d,tpr:%f,fpr:%f"%(f1,err_net,tp,fn,fp,tn,tpr,fpr))
    (f1,err_net,tp,fn,fp,tn,tpr,fpr) =np.mean([precision[i][1] for i in range(tests)],0)
    kn_un_precision = (f1,err_net,tp,fn,fp,tn,tpr,fpr)
    print("kn_un_precision: f1:%f,err_net:%d,tp:%d,fn:%d,fp:%d,tn:%d,tpr:%f,fpr:%f"%(f1,err_net,tp,fn,fp,tn,tpr,fpr))
    (f1,err_net,tp,fn,fp,tn,tpr,fpr) = np.mean([precision[i][2] for i in range(tests)],0)
    un_un_precision = (f1,err_net,tp,fn,fp,tn,tpr,fpr)
    print("un_un_precision: f1:%f,err_net:%d,tp:%d,fn:%d,fp:%d,tn:%d,tpr:%f,fpr:%f"%(f1,err_net,tp,fn,fp,tn,tpr,fpr))
    
    return index_order,auc_net,un_precision,kn_un_precision,un_un_precision

def part_constructor_evaluator_sgm(generator,tests,obj_matrix,sz,del_num):
    kn_nodes = sz-del_num
    precision = []
    
    kn_mask,un_mask,un_un_mask,kn_un_mask = partial_mask(sz,del_num)
    pre_adj = gumbel_p(generator,sz,un_mask).detach()
    index_order,P = sgraphmatch(obj_matrix,pre_adj,kn_nodes,iteration=20)
    pre_adj = torch.mm(torch.mm(P,pre_adj),P.T)
    print("index_order",index_order)
    auc_net = aucs(pre_adj,obj_matrix,un_mask,un_un_mask,kn_un_mask)
    
    
    print("auc:%f,un_un_auc:%f,kn_un_auc:%f"%(auc_net[0],auc_net[1],auc_net[2]))
    
    for t in range(tests):
        obj_matrix = obj_matrix.cuda()
        out_matrix = generator.sample_all(hard = True)#使用hard采样 才能计算tpr 与fpr   
        
        index_order,P = sgraphmatch(obj_matrix,out_matrix,kn_nodes,iteration=20)
        out_matrix = torch.mm(torch.mm(P,out_matrix),P.T)
        
        metrics_all = tpr_fpr_part(out_matrix,obj_matrix,un_mask,kn_un_mask,un_un_mask,del_num)
        precision.append(metrics_all)
    
    (f1,err_net,tp,fn,fp,tn,tpr,fpr) = np.mean([precision[i][0] for i in range(tests)],0)
    un_precision = (f1,err_net,tp,fn,fp,tn,tpr,fpr)
    print("f1:%f,err_net:%d,tp:%d,fn:%d,fp:%d,tn:%d,tpr:%f,fpr:%f"%(f1,err_net,tp,fn,fp,tn,tpr,fpr))
    (f1,err_net,tp,fn,fp,tn,tpr,fpr) =np.mean([precision[i][1] for i in range(tests)],0)
    kn_un_precision = (f1,err_net,tp,fn,fp,tn,tpr,fpr)
    print("kn_un_precision: f1:%f,err_net:%d,tp:%d,fn:%d,fp:%d,tn:%d,tpr:%f,fpr:%f"%(f1,err_net,tp,fn,fp,tn,tpr,fpr))
    (f1,err_net,tp,fn,fp,tn,tpr,fpr) = np.mean([precision[i][2] for i in range(tests)],0)
    un_un_precision = (f1,err_net,tp,fn,fp,tn,tpr,fpr)
    print("un_un_precision: f1:%f,err_net:%d,tp:%d,fn:%d,fp:%d,tn:%d,tpr:%f,fpr:%f"%(f1,err_net,tp,fn,fp,tn,tpr,fpr))
    
    return index_order,auc_net,un_precision,kn_un_precision,un_un_precision
    
    
    
    

# 计算离散序列状态的准确率
def accu(x_un,x_un_pre):
    return abs(torch.argmax(x_un,2)-torch.argmax(x_un_pre,2)).float().mean().cpu().detach().numpy()

# 已知和未知部分进行标记
def partial_mask(sz,del_num):
    '''mask the known part and un know part'''
    kn_mask = torch.zeros(sz,sz)
    kn_mask[:-del_num,:-del_num] = 1
    
    un_un_mask = torch.zeros(sz,sz)
    un_un_mask[-del_num:,-del_num:] = 1
    un_un_mask = un_un_mask - torch.diag(torch.diag(un_un_mask))
    
    left_mask = torch.ones(sz,sz)
    left_mask[:-del_num,:-del_num] = 0
    left_mask = left_mask-torch.diag(torch.diag(left_mask))
    kn_un_mask = left_mask - un_un_mask
    
    return kn_mask,left_mask,un_un_mask,kn_un_mask