import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.optim as optim
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
import numpy as np

def kronecker(A,B):
    return torch.einsum("ab,cd->acbd", A, B).view(A.size(0)*B.size(0),  A.size(1)*B.size(1))
class kronecker_Generator(nn.Module):
    def __init__(self,p0,korder = 3,node_num = 2):
        super(kronecker_Generator,self).__init__()
        self.p = Parameter(p0,requires_grad = True)
        # self.p = Parameter(torch.rand(node_num,node_num,requires_grad=True))
        self.korder = korder
        # print(self.p)
    def generator_adjacency(self):
        k = self.korder
        p0 = self.p
        adj = self.p
        for i in range(k-1):
            adj = kronecker(adj,p0)
        return adj
def loss_func(sigma,Pk):
    loss = -torch.sum((1-sigma)*torch.log(1-Pk)+sigma*torch.log(Pk))
    return loss
def metropolis_update_ratio(sigma_before,sigma_later,Pk):
    '''
    if memory is sufficinet, this one is much cleaner to execute
    '''
    Nll_before=(1-sigma_before)*np.log(1-Pk) +sigma_before*np.log(Pk)
    Nll_later=(1-sigma_later)*np.log(1-Pk) +sigma_later*np.log(Pk)
    ratio=np.exp(np.sum(Nll_later-Nll_before))
    return ratio
def SwapElement(sigma_before,i,j):
    i_topology=sigma_before[i,:]
    j_topology=sigma_before[j,:]
    sigma_later=np.copy(sigma_before)
    sigma_later[i,:]=j_topology
    sigma_later[j,:]=i_topology
    sigma_later[:,i]=sigma_before[:,j]
    sigma_later[:,j]=sigma_before[:,i]
    sigma_later[i,j]=sigma_before[i,j]
    sigma_later[j,i]=sigma_before[j,i]
    sigma_later[i,i]=sigma_before[j,j]
    sigma_later[j,j]=sigma_before[i,i]
    return sigma_later
def SamplePermutation(Pk,sigma,u,n1_swap,n2_swap,label_non_obs):
    sigma_later=SwapElement(sigma,n1_swap, n2_swap)
    ratio=metropolis_update_ratio(sigma,sigma_later,Pk)
    ifswap = False
    if u<ratio:
      sigma=sigma_later
      label_non_obs=SwapElement(label_non_obs,n1_swap, n2_swap)
      ifswap = True
    return sigma,label_non_obs,ifswap
def SampleZ(H,Pk,label_non_obs,u):
    mat_size=len(Pk)
    edge_in_non_obs=(H>0)*label_non_obs  
    edge_position=np.where(edge_in_non_obs)
    edge_removed=np.random.randint(len(edge_position[0]))
    # print("edge_removed",edge_removed,edge_position[0][edge_removed],edge_position[1][edge_removed])
    px=Pk[edge_position[0][edge_removed],edge_position[1][edge_removed]]
    non_edge_in_non_obs=(H<1)*label_non_obs
    ##return 
    py_array=non_edge_in_non_obs*Pk
    py_array=np.ravel(py_array)
    py=np.random.choice(range(len(py_array)), size=1, p=py_array/np.sum(py_array))
    ratio=(1-py_array[py])/(1-px)
    if ratio<u:
        H[py//mat_size,py%mat_size]=1
        H[edge_position[0][edge_removed],edge_position[1][edge_removed]]=0
        # print('accept')
    return H

def missing_label(sz,missing_percent):
     # random sample
    missing_num = int(sz*missing_percent)
    idx = torch.randperm(sz)[:missing_num]
    mask_un_obs = torch.zeros(sz,sz)
    for i in idx:
        mask_un_obs[i] = 1
        mask_un_obs[:,i:i+1] = 1
    mask_obs = 1- mask_un_obs
    return mask_un_obs,mask_obs,idx
# only sigma and kronecker pk are itered
def E_step(sigma,N,warmup,Pk,label_non_obs,perm):
    u1=np.random.rand(N+warmup)
    u2=np.random.rand(N+warmup)
    sigma_hist=[]
    Z_label=[]
    Node_list=np.arange(len(Pk))
    element_to_swap=np.random.choice(a=Node_list,size=(2,3*(N+warmup)))
    mask=element_to_swap[1,:]!=element_to_swap[0,:]# it is pointless to swap the same element
    n1_swap=element_to_swap[0,:][mask]
    n2_swap=element_to_swap[1,:][mask]
   
    for i in range(N+warmup):
        # if i%100==0:
        #     print("E_step",i,round(i/(N+warmup),4))
        sigma=SampleZ(sigma,Pk,label_non_obs,u1[i]) # sigma shuffle后的adj
        # print("n1,n2",n1_swap,n2_swap,mask)
        sigma,label_non_obs,ifswap=SamplePermutation(Pk,sigma,u2[i],n1_swap[i],n2_swap[i],label_non_obs)
        if ifswap:
            perm[n1_swap[i]],perm[n2_swap[i]] = perm[n2_swap[i]],perm[n1_swap[i]]
            print(ifswap,n1_swap[i],n2_swap[i])
        if i>=warmup:
            sigma_hist.append(sigma)
            Z_label.append(label_non_obs)
    return sigma_hist,Z_label,perm #（H+G）

def M_step(epoch,sigma_train,p0,k,N):
    losses = []
    generator = kronecker_Generator(p0,k,2)
    learning_rate = 1e-5#0.0000001
    opt_net = optim.SGD(generator.parameters(),lr = learning_rate)
    decayRate = 0.95

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt_net, gamma=decayRate) 
    for i in range(epoch):   
        opt_net.zero_grad()
        Pk = generator.generator_adjacency()
        loss = loss_func(sigma_train,Pk,N)
        loss2 = loss2_func(sigma_train.detach().numpy(),Pk.detach().numpy(),N)
        loss.backward()
        losses.append(loss.item())
        # print(str(i),loss.item(),loss2.item(),(loss2-loss).item())
        opt_net.step()
        scheduler.step()
        for group in opt_net.param_groups:
            for param in group["params"]: 
              # print("before param",param)
              param.data.clamp_(0.0001,0.9999)
              # print("after param",param)
    
        for p in generator.parameters():
            p0 = p.data  
        generator = kronecker_Generator(p0,k,2)
        Pk = generator.generator_adjacency()
        Pk = Pk.detach().numpy() 
        # evaluation
        #   nll infer  nll true
    return np.mean(losses),Pk,p0

def kronEM(iterstep,N,warmup,H,Pk,label_non_obs,epoch,p0,perm,k,obj_adj,init_label_non_obs):
    emlosses = []
    for i in range(iterstep):
        print("start iterstep",i)
        print("diff1",abs((H - obj_adj)*(1-label_non_obs)).sum())
        sigma_hist,Z_label,perm = E_step(H,N,warmup,Pk,label_non_obs,perm)
        print("E-step")
        H = sigma_hist[-1]
        label_non_obs = Z_label[-1]
        sigma_hist_train = np.array(sigma_hist)
        sigma_train = torch.DoubleTensor(sigma_hist_train)
        emloss,Pk,p0 = M_step(epoch,sigma_train,p0,k,N)
        emlosses.append(emloss)
        print("*************\n EM loss is %f"%emloss,"p0 is ",p0)
        # evaluation
        inferp_nll = NLL(H,Pk)
        # truep_nll =NLL(H,ground_adj.detach().numpy())
        print("inferp_nll is %f "%(inferp_nll))
        obs_mask = (1-label_non_obs).astype(bool)
        obs_auc = calauc(H,Pk,obs_mask)
        non_obs_auc = calauc(H,Pk,label_non_obs.astype(bool))
        print(perm)
        permutation_mat  = np.eye(H.shape[0])[perm]
        shuffle_H = np.dot(np.dot(permutation_mat,H),H.T)
        label_obs = (1-init_label_non_obs)
        print("label_non",abs(init_label_non_obs-label_non_obs).sum())

        obs_diff_edge = abs((H  - obj_adj)*label_obs).sum()
        unobs_diff_edge =abs(init_label_non_obs*(shuffle_H - obj_adj)).sum()
        all_diff_edge = abs(shuffle_H - obj_adj).sum()
        # D_abs = torch.abs(orip-p0).mean()
        print("infer p is ",p0,  "obseved auc  is %f and non_obs auc is %f and obs_diff_edge is %f and un_obs_diff_edge %f and all_diff_edge is %f"%(obs_auc,non_obs_auc,obs_diff_edge,unobs_diff_edge,all_diff_edge))
    return emlosses,perm

def NLL(sigma_true,pk):
  Nll_before=(1-sigma_true)*np.log(1-pk) +sigma_true*np.log(pk)
  return -np.sum(Nll_before)

def loss2_func(sigma_train,Pk,N):
  loss2 = 0
  for i in range(N):
    loss2+= NLL(sigma_train[i],Pk)
    # print(loss2)
  return loss2/N

def loss_func(sigma,Pk,N):
    loss = -torch.sum((1-sigma)*torch.log(1-Pk)+sigma*torch.log(Pk))
    return loss/N

def generator_adj(korder,p):
    k = korder
    p0 = p
    adj = p
    for i in range(k-1):
        adj = kronecker(adj,p0)
    return adj
def calauc(H,Pk,mask):
  fpr, tpr, thresholds = roc_curve(H[mask],Pk[mask])
  Auc = auc(fpr, tpr)
  return Auc
