import torch
import time
import random
import torch.nn.functional as F
import torch.nn.utils as U
import torch.optim as optim
import argparse
import sys 
import pickle
import matplotlib.pyplot as plt
from model_gy import *
from tools_changedelmethod import *
import os


parser = argparse.ArgumentParser(description = "ER network")
parser.add_argument('--node_num', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument("--seed",type =int,default = 135,help = "random seed (default: 2050)")
# parser.add_argument("--sysdyn",type = str,default = 'voter',help = "the type of dynamics")
parser.add_argument("--dim",type = int,default = 2,help = "information diminsion of each node cml")
parser.add_argument("--hidden_size",type = int,default = 64,help = "hidden size of GGN model (default:128)")
parser.add_argument("--epoch_num",type = int,default = 700,help = "train epoch of model (default:1000)")                    
parser.add_argument("--batch_size",type = int,default = 1024,help = "input batch size for training (default: 128)")
parser.add_argument("--cuda",type = int, default = 3,help = "choose the GPU (default: 0)")
parser.add_argument("--lr_net",type = float,default = 0.004,help = "gumbel generator learning rate (default:0.004) ")
parser.add_argument("--lr_dyn",type = float,default = 0.001,help = "dynamic learning rate (default:0.001)")
parser.add_argument("--lr_state",type = float,default = 0.1,help = "state learning rate (default:0.1)")
parser.add_argument("--miss_percent",type = float,default = 0.1,help = "missing percent node (default:0.1)")
parser.add_argument("--data_path",type =str,default = "/data/chenmy/voter/seed2050ba1005000",help = "the path of simulation data (default:ER_p0.04100300) ")
args = parser.parse_args()
# from model_old_generator import *
# configuration
HYP = {
    'node_num': args.node_num,  # node num
    'seed': args.seed,  # the seed
    'dim': args.dim,  # information diminsion of each node cml:1 spring:4
    'hid': args.hidden_size,  # hidden size
    'epoch_num': args.epoch_num,  # epoch
    'batch_size': args.batch_size,  # batch size
    'lr_net':args.lr_net,  # lr for net generator
    'lr_dyn': args.lr_dyn,  # lr for dyn learner
    'lr_state':args.lr_state,
    'miss_percent':args.miss_percent,
    'data_path':args.data_path,
    'temp': 1,  # 温度
    'drop_frac': 1,  # temperature drop frac
    "cuda":args.cuda,
}

print("all parameter ",HYP)
# partial known adj 
torch.cuda.set_device(HYP["cuda"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(HYP['seed'])
random.seed(HYP['seed'])
del_num = int(args.node_num*args.miss_percent)
kn_nodes = HYP['node_num'] - del_num
filename = os.path.abspath(__file__)
print("filename",filename)


num = 3
def onehot_state(x_un_pre):
    if x_un_pre.shape[1]==1:
        state = torch.argmax(x_un_pre,2)
        pre_state = torch.cat((state,1-state),1).unsqueeze(1)
    else:
        state = torch.argmax(x_un_pre,2).unsqueeze(2)
        pre_state = torch.cat((state,1-state),2)  
    return pre_state

# load data
adj_address = HYP['data_path']+'-adjmat.pickle'
series_address = HYP['data_path']+'-series.pickle'
# train_loader, val_loader, test_loader, object_matrix = load_bn_ggn_ori(series_address,adj_address,batch_size=HYP['batch_size'])
train_loader, val_loader, test_loader, object_matrix = load_bn_ggn(series_address,adj_address,batch_size=HYP['batch_size'],seed = HYP['seed'])
print(object_matrix[-del_num:,-del_num:,].sum(0))

# check  未知与未知部分的link 
unedges  = torch.sum(object_matrix[-del_num:,-del_num:])
while unedges.item() == 0:
    print("sample 0 edges")
    sys.exit()

start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('start_time:', start_time)
    
# dyn learner isomorphism
dyn_isom = IO_B_discrete(HYP['dim'], HYP['hid']).to(device)
op_dyn = optim.Adam(dyn_isom.parameters(), lr=HYP['lr_dyn'])

# net learner 

generator = Gumbel_Generator_nc(sz=HYP['node_num'],del_num = del_num,temp=HYP['temp'], temp_drop_frac=HYP['drop_frac']).to(device)
generator.init(0, 0.1)
op_net = optim.Adam(generator.parameters(), lr=HYP['lr_net'])

# states learner
sample_num = len(train_loader.dataset)
states_learner = Generator_states_discrete(sample_num,del_num).double()
if use_cuda:
    states_learner = states_learner.cuda()
opt_states = optim.Adam(states_learner.parameters(),lr = HYP['lr_state'])
train_index = DataLoader([i for i in range(sample_num)],HYP['batch_size'])

# val  states learner 
v_sample_num = len(val_loader.dataset)
states_learner_v = Generator_states_discrete(v_sample_num,del_num).double()
if use_cuda:
    states_learner_v = states_learner_v.cuda()
opt_states_v = optim.Adam(states_learner_v.parameters(),lr = HYP['lr_state'])
val_index = DataLoader([i for i in range(v_sample_num)],HYP['batch_size'])


observed_adj = object_matrix[:-del_num,:-del_num]
kn_mask,left_mask,un_un_mask,kn_un_mask = partial_mask(HYP['node_num'],del_num)
print("data_num",len(train_loader.dataset),"object_matrix",object_matrix)
loss_fn = torch.nn.NLLLoss()


# 训练全网的正向动力学
'''voter states,dyn,NET 同时训练 '''
# indices to draw the curve

num = 0
def train_dyn_net_state():
    loss_batch = []
    ymae_batch = []
    for idx,(data,states_id) in enumerate(zip(train_loader,train_index)):
        x = data[0].float().to(device)
        y = data[1].to(device).long()

        x_kn = x[:,:-del_num,:]
        y_kn = y[:,:-del_num]
    
        x_un = x[:,-del_num:,:]
        generator.drop_temp()
        outputs = torch.zeros(y.size(0), y.size(1),2)
        loss_node = []
        ymae_node = []
        
        for j in range(HYP['node_num']-del_num):
            op_net.zero_grad()
            op_dyn.zero_grad()
            opt_states.zero_grad()
            
            x_un_pre = states_learner(states_id.cuda())
            
            x_hypo = torch.cat((x_kn,x_un_pre.float()),1)
    
            adj_col = generator.sample_all()[:,j]
            adj_col[:-del_num] = observed_adj[j].cuda()

            
            y_hat = dyn_isom(x_hypo, adj_col, j, num, HYP['node_num']-del_num)
            loss = loss_fn(y_hat,y[:,j])
            loss.backward()

            mae = torch.mean(abs(y[:,j] - torch.argmax(y_hat,1)).float())  

            # cut gradient in case nan shows up
            U.clip_grad_norm_(generator.gen_matrix, 0.000075)

            op_dyn.step()
            op_net.step()
            opt_states.step()

            # use outputs to caculate mse
            outputs[:, j, :] = y_hat
            
            # record
            ymae_node.append(mae.item())
            loss_node.append(loss.item())

        ymae_batch.append(np.mean(ymae_node))
        loss_batch.append(np.mean(loss_node))
        
    return np.mean(loss_batch),np.mean(ymae_batch)

def val():
    loss_batch = []
    ymae_batch = []
    choose_node = torch.randint(0,HYP['node_num']-del_num,[choose_num])
    hypo_adj = generator.sample_all().detach()
    hypo_adj[:-del_num,:-del_num] = observed_adj
    for (idx,data),state_id in zip(enumerate(val_loader),val_index):
        
        x = data[0].float().to(device)
        y = data[1].to(device).long()
        
        x_kn = x[:,:-del_num,:]
        y_kn = y[:,:-del_num]
    
        x_un = x[:,-del_num:,:]
        generator.drop_temp()
        outputs = torch.zeros(y.size(0), y.size(1),2)
        loss_node = []
        states_node = []

        for index,j in enumerate(choose_node):

            x_un_pre = states_learner_v(state_id.cuda())
            x = torch.cat((x_kn,x_un_pre.float()),1)
            
            opt_states_v.zero_grad()
            adj_col = hypo_adj[:,j].cuda()# hard = true
            y_hat = dyn_isom(x, adj_col, j, num, HYP['node_num'])
            
            loss = loss_fn(y_hat,y[:,j])
            loss.backward()
            opt_states_v.step()
            ymae = torch.mean(abs(y[:,j] - torch.argmax(y_hat,1)).float())  
            # states_node
            outputs[:, index, :] = y_hat
            
            # record
            loss_node.append(loss.item())            
            states_node.append(ymae.item())

        ymae_batch.append(np.mean(states_node))
        loss_batch.append(np.mean(loss_node))
        
    return torch.mean(torch.FloatTensor(loss_batch)),torch.mean(torch.FloatTensor(ymae_batch))

# train
val_epoch =  100
choose_num  = int(HYP['node_num']/3)
losses = []
val_loss_epoch = []
metric_epoch_yt=[]

for epoch in range(HYP['epoch_num']):
    start_time = time.time()
    loss_y,mse_y = train_dyn_net_state()
    (index_order,auc_net,precision,kn_un_precision,un_un_precision) = part_constructor_evaluator_sgm(generator,1,object_matrix,HYP["node_num"],del_num)
    metric_epoch_yt.append([auc_net,precision,kn_un_precision,un_un_precision])
    losses.append([loss_y,mse_y])
    if auc_net[0]>0.95:
        break
    print(epoch,'gumbel all Net error: %f,kn_un :%f,un_un:%f'%(round(float(precision[1].item()/left_mask.sum()),2),round(float(kn_un_precision[1].item()/kn_un_mask.sum()),2),round(float(un_un_precision[1].item()/un_un_mask.sum()),2)))   
    print("index_order",index_order)
    print('loss_y:%f,mse_y:%f'%(loss_y,mse_y))
    end_time = time.time()
    print("cost_time",str(round(end_time -start_time, 2)))

    if (epoch+1)%100 ==0:
        print("\nstatred val",epoch)
        val_losses = [];val_maes = []
        for i in range(val_epoch):
            val_loss,val_mae = val()
            val_losses.append(val_loss)
            val_maes.append(val_mae)
        vloss = torch.mean(torch.FloatTensor(val_losses))
        vmae = torch.mean(torch.FloatTensor(val_maes))
        val_loss_epoch.append([vloss,vmae])
        print('     val loss:' + str(vloss)+' val mse:' + str(vmae),'\n')#
end_time = time.time()
print("cost_time",str(round(end_time -start_time, 2)))