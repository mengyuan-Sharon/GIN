# %load train_new.py
import torch
import time
import sys
import random
import torch.nn.functional as F
import torch.nn.utils as U
import torch.optim as optim
import argparse
import matplotlib.pyplot as plt

from model_gy import *
from tools_changedelmethod import *

parser = argparse.ArgumentParser(description = "ER network")
parser.add_argument('--node_num', type=int, default=100,
                    help='number of epochs to train')#'--epoch_num'
parser.add_argument('--node_size', type=int, default=100,
                    help='number of epochs to train')
parser.add_argument("--seed",type =int,default = 2050,help = "random seed (default: 2050)")
parser.add_argument("--dim",type = int,default = 1,help = "information diminsion of each node cml")
parser.add_argument("--hidden_size",type = int,default = 128,help = "hidden size of GGN model (default:128)")
parser.add_argument("--epoch_num",type = int,default = 1000,help = "train epoch of model (default:500)")                    
parser.add_argument("--batch_size",type = int,default = 1024,help = "input batch size for training (default: 128)")
parser.add_argument("--lr_net",type = float,default = 0.004,help = "gumbel generator learning rate (default:0.004) ")
parser.add_argument("--lr_dyn",type = float,default = 0.001,help = "dynamic learning rate (default:0.001)")
parser.add_argument("--lr_stru",type = float, default = 0.0000001,help = "sparse network adjustable rate (default:0.000001)" )
parser.add_argument("--lr_state",type = float,default = 0.1,help = "state learning rate (default:0.1)")
parser.add_argument("--miss_percent",type = float,default = 0.1,help = "missing percent node (default:0.1)")
parser.add_argument("--data_path",type =str,default = "/data/chenmy/cml/cmlER-100-2500.pickle",help = "the path of simulation data (default:ER_p0.04100300) ") #/data/zhangyan/cml/data_100_er_10000.picklecmler300_30000
args = parser.parse_args()

# print("no lr stru")
# from model_old_generator import *
# configuration
HYP = {
    'note': 'try init',
    'node_num': args.node_num,  # node num
    'node_size': args.node_size, # node size
    'conn_p': '25',  # connection probility : 25 means 1/25
    'seed': args.seed,  # the seed
    'dim': args.dim,  # information diminsion of each node cml:1 spring:4
    'hid': args.hidden_size,  # hidden size
    'epoch_num': args.epoch_num,  # epoch
    'batch_size': args.batch_size,  # batch size
    'lr_net':args.lr_net,  # lr for net generator
    'lr_dyn': args.lr_dyn,  # lr for dyn learner
    'lr_stru': args.lr_stru,  # lr for structural loss 0.0001
    'lr_state':args.lr_state,
    'hard_sample': False,  # weather to use hard mode in gumbel
    'sample_time': 1,  # sample time while training
    'sys': 'cml',  # simulated system to model
    'isom': True,  # isomorphic, no code for sync
    'temp': 1,  # 温度
    'drop_frac': 1,  # temperature drop frac
    'save': True, # weather to save the result
}


print("has structure parameter:",HYP)
# partial known adj 
del_num = int(args.node_num*args.miss_percent)
print("del_num",del_num)


cuda_number = 3
torch.set_num_threads(1)
torch.cuda.set_device(cuda_number)
print('cuda',cuda_number)
torch.manual_seed(HYP['seed'])  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 多步
# hid size
# 权值共享
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('start_time:', start_time)
rec_res = record_stop(start_time, HYP)
if rec_res == True:
    print('record success')
else:
    print('record failed')
    d()


# dyn learner isomorphism
dyn_isom = IO_B(HYP['dim'], HYP['hid']).to(device)
# optimizer
op_dyn = optim.Adam(dyn_isom.parameters(), lr=HYP['lr_dyn'])

# net learner 

generator = Gumbel_Generator_nc(sz=HYP['node_num'],del_num = del_num,temp=HYP['temp'], temp_drop_frac=HYP['drop_frac']).to(device)
generator.init(0, 0.1)
op_net = optim.Adam(generator.parameters(), lr=HYP['lr_net'])


data_path = args.data_path
print("data_path",data_path)

# load data
train_loader, val_loader, test_loader, object_matrix = load_cml_ggn(data_path,batch_size=HYP['batch_size'],
                                                                        node=HYP['node_num'])
# states learner
sample_num = len(train_loader.dataset)
print("data_num:",sample_num)
states_learner = Generator_states(sample_num,del_num).double()
if use_cuda:
    states_learner = states_learner.cuda()
opt_states = optim.Adam(states_learner.parameters(),lr = HYP['lr_state'])
train_index = DataLoader([i for i in range(sample_num)],HYP['batch_size'])
observed_adj = object_matrix[:-del_num,:-del_num]
kn_mask,left_mask,un_un_mask,kn_un_mask = partial_mask(HYP['node_num'],del_num)
# val_states learner 
v_sample_num = len(val_loader.dataset)
states_learner_v = Generator_states(v_sample_num,del_num).double()
if use_cuda:
    states_learner_v = states_learner_v.cuda()
opt_states_v = optim.Adam(states_learner_v.parameters(),lr = HYP['lr_state'])
val_index = DataLoader([i for i in range(v_sample_num)],HYP['batch_size'])
observed_adj = object_matrix[:-del_num,:-del_num]
kn_mask,left_mask,un_un_mask,kn_un_mask = partial_mask(HYP['node_num'],del_num)
un_edges = object_matrix[-del_num:,-del_num:].sum()
while un_edges == 0:
    sys.exit()

num = int(HYP['node_num'] / HYP['node_size'])
remainder = int(HYP['node_num'] % HYP['node_size'])
if remainder == 0:
    num = num - 1
kn_nodes = HYP['node_num'] - del_num

'''states,dyn,NET 同时训练 '''
def train_dyn_gen_state():
    loss_batch = []
    mse_batch = []
    
    for data,states_id in zip(train_loader,train_index):
        # print('batch idx:', idx)
        # data
        data = data.to(device)
        x = data[:, :-del_num, 0, :]
        y = data[:, :-del_num, 1, :]
        x_un = data[:, -del_num:, 0, :]

        # drop temperature
        generator.drop_temp()
        
        outputs = torch.zeros(y.size(0), y.size(1), y.size(2)).cuda()
        
        loss_node = []
        states_node = []
        for j in range(HYP['node_num']-del_num):
            
            # zero grad
            op_net.zero_grad()
            op_dyn.zero_grad()
            opt_states.zero_grad()
            
            x_un_pre = states_learner(states_id.cuda())#.detach() # 固定states 状态
            x_hypo = torch.cat((x,x_un_pre.float()),1)
    
            hypo_adj = generator.sample_all()
            hypo_adj[:-del_num,:-del_num] = observed_adj
        

            adj_col = hypo_adj[:,j].cuda()# hard = true
             
                
            y_hat = dyn_isom(x_hypo, adj_col, j, num, HYP['node_size'])
            

            loss = torch.mean(torch.abs(y_hat - y[:, j, :]))# +abs(adj_col.sum()-4)# 绝对值做loss
            
            #print(j,loss.item())
            # states_node
            
            # backward and optimize
            loss.backward()
            
            # cut gradient in case nan shows up
            U.clip_grad_norm_(generator.gen_matrix, 0.000075)
            
            op_dyn.step()
            op_net.step()
            opt_states.step()

            # use outputs to caculate mse
            outputs[:, j, :] = y_hat

            # record
            loss_node.append(loss.item())

        # dif_batch.append(np.mean(dif_node))
        loss_batch.append(torch.mean(torch.FloatTensor(loss_node)))
        mse_batch.append(F.mse_loss(y,outputs).item()) 
        
    # used for more than 10 nodes
    #op_net.zero_grad()
    #loss = (torch.sum(generator.sample_all())) * HYP['lr_stru']  # 极大似然
    #loss.backward()
    #op_net.step()
    
    return torch.mean(torch.FloatTensor(loss_batch)),torch.mean(torch.FloatTensor(mse_batch))

def cal_states_loss():
    mae =[];mse = [];
    for data,states_id in zip(train_loader,train_index):
        data = data.to(device)
        x = data[:, :-del_num, 0, :]
        y = data[:, :-del_num, 1, :]
        x_un = data[:, -del_num:, 0, :]
        
        x_un_pre = states_learner(states_id.cuda()).detach()
        x_un_pre_p = torch.index_select(x_un_pre,1,index_order.long())
        
        mae.append(torch.mean(abs(x_un.cuda()-x_un_pre_p)).item())
        mse.append(F.mse_loss(x_un.float(),x_un_pre_p.float()).item())  
        # break
    return  torch.mean(torch.FloatTensor(mae)),torch.mean(torch.FloatTensor(mse))

def val():
    loss_batch = []
    mse_batch = []
    choose_node = torch.randint(0, HYP['node_num'],[choose_num])
    hypo_adj = generator.sample_all().detach()
    hypo_adj[:-del_num,:-del_num] = observed_adj
    for (idx,data),state_id in zip(enumerate(val_loader),val_index):
        data = data.to(device)
        x_kn = data[:, :-del_num, 0, :]
        y = data[:, :, 1, :]
        y_c = torch.index_select(y,1,choose_node.cuda())
        outputs  = torch.zeros_like(y_c)
        # drop temperature
        generator.drop_temp()
        # outputs = torch.zeros(y.size(0), y.size(1), y.size(2))
        loss_node = []
        states_node = []

        for index,j in enumerate(choose_node):
            x_un_pre = states_learner_v(state_id.cuda())
            x = torch.cat((x_kn,x_un_pre.float()),1)
            opt_states_v.zero_grad()
            adj_col = hypo_adj[:,j].cuda()# hard = true
            y_hat = dyn_isom(x, adj_col, j, num, HYP['node_num'])
            loss = torch.mean(torch.abs(y_hat - y[:, j, :]))# +abs(adj_col.sum()-4)# 绝对值做loss
            # print("loss:",loss)
            loss.backward()
            opt_states_v.step()
            # states_node
            outputs[:, index, :] = y_hat
            # record
            loss_node.append(loss.item())
        # dif_batch.append(np.mean(dif_node))
        loss_batch.append(np.mean(loss_node))
        mse_batch.append(F.mse_loss(y_c,outputs).item())
    return torch.mean(torch.FloatTensor(loss_batch)),torch.mean(torch.FloatTensor(mse_batch))

'''states,dyn,NET 同时训练 '''
choose_num =100
val_epoch = 50
# 首先训练正向动力学
loss_epoch = []
metric_epoch = []
states_loss = []
val_loss_epoch = []
begin_time = time.time()
for epoch in range(HYP['epoch_num']):
    start_time = time.time()
    loss,mse = train_dyn_gen_state()
    (index_order,auc_net,precision,kn_un_precision,un_un_precision) = part_constructor_evaluator_sgm(generator,1,object_matrix,HYP["node_num"],del_num)
    smae,smse = cal_states_loss()
    states_loss.append([smae,smse])
    loss_epoch.append([loss,mse])
    metric_epoch.append([auc_net,precision,kn_un_precision,un_un_precision])
    print(epoch,'gumbel all Net error: %f,kn_un :%f,un_un:%f'%(round(float(precision[1].item()/left_mask.sum()),2),round(float(kn_un_precision[1].item()/kn_un_mask.sum()),2),round(float(un_un_precision[1].item()/un_un_mask.sum()),2)))   
    print('loss:' + str(loss)+' mse:' + str(mse),"states mae mse",smae,smse)#
    end_time = time.time()
    print("cost_time",str(round(end_time -start_time, 2)))
    if epoch%100 ==0:
        print("\nstatred val")
        val_losses = [];val_mses = []
        for i in range(val_epoch):
            val_loss,val_mse=  val()
            val_losses.append(val_loss)
            val_mses.append(val_mse)
        vloss = torch.mean(torch.FloatTensor(val_losses))
        vmse = torch.mean(torch.FloatTensor(val_mses))
        val_loss_epoch.append([vloss,vmse])
        print('val loss:' + str(vloss)+' val mse:' + str(vmse))#

over_time = time.time()
print("all cost time ",str(round(over_time -begin_time, 2)))
save_path = '/home/chenmy/scalenetwork/scalesaved/'+str(args.node_num)+str(time.strftime("%Y%m%d%H:%M:%S",time.localtime(end_time)))+'.pickle'
print(save_path)
results = [loss_epoch,states_loss,metric_epoch]#
with open(save_path, 'wb') as f:
    pickle.dump(results, f)

dyn_savepath =  '/home/chenmy/scalenetwork/scalesaved/'+"dyn"+str(args.node_num)+str(time.strftime("%Y%m%d%H:%M:%S",time.localtime(time.time())))+'.dyn'
print(dyn_savepath)
torch.save(generator.state_dict(),dyn_savepath)
# 存储gumbel model
gumbel_savepath =  '/home/chenmy/scalenetwork/scalesaved/'+"gumbelre"+str(args.node_num)+str(time.strftime("%Y%m%d%H:%M:%S",time.localtime(end_time)))+'.gt'
print(gumbel_savepath)
torch.save(generator.state_dict(),gumbel_savepath)
state_savepath =  '/home/chenmy/scalenetwork/scalesaved/'+"states_learn"+str(args.node_num)+str(time.strftime("%Y%m%d%H:%M:%S",time.localtime(end_time)))+'.sta'
print(state_savepath)
torch.save(states_learner.state_dict(),state_savepath)