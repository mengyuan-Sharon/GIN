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
import seaborn as sns
from model_gy import *
from tools_changedelmethod import *

parser = argparse.ArgumentParser(description = "ER network")
parser.add_argument('--node_num', type=int, default=115,
                    help='number of epochs to train')
parser.add_argument('--node_size', type=int, default=115,
                    help='number of epochs to train')
parser.add_argument("--seed",type =int,default = 157,help = "random seed (default: 2050)")
parser.add_argument("--sysdyn",type = str,default = 'voter',help = "the type of dynamics")
parser.add_argument("--dim",type = int,default = 2,help = "information diminsion of each node cml")
parser.add_argument("--hidden_size",type = int,default = 64,help = "hidden size of GGN model (default:128)")
parser.add_argument("--epoch_num",type = int,default = 300,help = "train epoch of model (default:1000)")                    
parser.add_argument("--batch_size",type = int,default = 1024,help = "input batch size for training (default: 128)")
parser.add_argument("--lr_net",type = float,default = 0.004,help = "gumbel generator learning rate (default:0.004) ")
parser.add_argument("--lr_dyn",type = float,default = 0.01,help = "dynamic learning rate (default:0.001)")
parser.add_argument("--lr_dyn_r",type = float,default = 0.01,help = "dyn reverse learning rate (default:0.1)")
parser.add_argument("--lr_stru",type = float, default = 0.00001,help = "sparse network adjustable rate (default:0.000001)" )
parser.add_argument("--lr_state",type = float,default = 0.05,help = "state learning rate (default:0.1)")
parser.add_argument("--miss_percent",type = float,default = 0.1,help = "missing percent node (default:0.1)")
parser.add_argument("--data_path",type =str,default = "./data/mark-renyi20020000",help = "the path of simulation data (default:ER_p0.04100300) ")

args = parser.parse_args()
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
    'lr_dyn_r':args.lr_dyn_r,
    'lr_stru': args.lr_stru,  # lr for structural loss 0.0001
    'lr_state':args.lr_state,
    'hard_sample': False,  # weather to use hard mode in gumbel
    'sample_time': 1,  # sample time while training
    'sys': args.sysdyn,  # simulated system to model
    'isom': True,  # isomorphic, no code for sync
    'temp': 1,  # 温度
    'drop_frac': 1,  # temperature drop frac
    'save': True, # weather to save the result
}

print("parameter",HYP)

# partial known adj 
torch.cuda.set_device(0)
torch.manual_seed(HYP['seed'])
del_num = int(args.node_num*args.miss_percent)
print(del_num)
# load data
#adj_address = '/home/chenmy/GGN_voter/data/voter/ER15000_100-adjmat.pickle'
#series_address = '/home/chenmy/GGN_voter/data/voter/ER15000_100-series.pickle'

# adj_address = '/data/chenmy/voter/seed2051ws15020000-adjmat.pickle'
# series_address = '/data/chenmy/voter/seed2051ws15020000-series.pickle'

# ba 5条link
# adj_address = '/data/chenmy/voter/seed2051ba15020000-adjmat.pickle'
# series_address = '/data/chenmy/voter/seed2051ba15020000-series.pickle'
# print("sgd",adj_address)

# football
# adj_address = '/data/chenmy/voter/seed2051football11520000-adjmat.pickle'
# series_address = '/data/chenmy/voter/seed2051football11520000-series.pickle'
adj_address = '/data/chenmy/voter/seed2051football11510000-adjmat.pickle'
series_address = '/data/chenmy/voter/seed2051football11510000-series.pickle'
# karate 
# adj_address = '/data/chenmy/voter/seed2051karate343000-adjmat.pickle'
# series_address = '/data/chenmy/voter/seed2051karate343000-series.pickle'

# adj_address = '/data/zhangyan/voter/mark-renyi10010000-adjmat.pickle'
# series_address = '/data/zhangyan/voter/mark-renyi10010000-series.pickle'
# adj_address = '/data/chenmy/voter/seed2051email14320000-adjmat.pickle'
# series_address = '/data/chenmy/voter/seed2051email14320000-series.pickle'
train_loader, val_loader, test_loader, object_matrix = load_bn_ggn(series_address,adj_address,batch_size=HYP['batch_size'],seed = HYP["seed"])
unedges  = torch.sum(object_matrix[-del_num:,-del_num:])
while unedges.item() == 0:
    print("sample 0 edges")
    sys.exit()


start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('start_time:', start_time)
rec_res = record_stop(start_time, HYP)


if rec_res == True:
    print('record success')
else:
    print('record failed')
    d()
  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dyn learner isomorphism

dyn_isom = IO_B_discrete(HYP['dim'], HYP['hid']).to(device)
op_dyn = optim.Adam(dyn_isom.parameters(), lr=HYP['lr_dyn'])

# dyn_reverse
dyn_reverse_r = IO_B_discrete(HYP['dim'], HYP['hid']).to(device)
op_dyn_r = optim.Adam(dyn_reverse_r.parameters(),lr = HYP['lr_dyn_r'])

# net learner 
generator = Gumbel_Generator_nc(sz=HYP['node_num'],del_num = del_num,temp=HYP['temp'], temp_drop_frac=HYP['drop_frac']).to(device)
generator.init(0, 0.1)
op_net = optim.Adam(generator.parameters(), lr=HYP['lr_net'])

# states learner
sample_num = len(train_loader.dataset)
if HYP["sys"] == "cml":
    states_learner = Generator_states(sample_num,del_num).double()
elif HYP["sys"] == "voter":
    states_learner = Generator_states_discrete(sample_num,del_num).double()
if use_cuda:
    states_learner = states_learner.cuda()
    
opt_states = optim.Adam(states_learner.parameters(),lr = HYP['lr_state'])
# opt_states = optim.SGD(states_learner.parameters(),lr = HYP['lr_state'])

train_index = DataLoader([i for i in range(sample_num)],HYP['batch_size'])

observed_adj = object_matrix[:-del_num,:-del_num]
kn_mask,left_mask,un_un_mask,kn_un_mask = partial_mask(HYP['node_num'],del_num)
infer_edges = torch.sum(object_matrix).item() - torch.sum(observed_adj).item()

print("data_num",len(train_loader.dataset),"object_matrix",object_matrix)
loss_fn = torch.nn.NLLLoss()

'''voter states,dyn,NET 同时训练 '''
# indices to draw the curve
def train_dyn_gen_state_voter():
    loss_batch = []
    statesmae_batch = []
    
    # print('current temp:',generator.temperature)
    # NUM_OF_1.append(torch.sum(generator.sample_all()))
    for idx,(data,states_id) in enumerate(zip(train_loader,train_index)):
        # print('batch idx:', idx)
        # data
        x = data[0].float().to(device)
        y = data[1].to(device).long()
        
        x_kn = x[:,:-del_num,:]
        y_kn = y[:,:-del_num]
        
        x_un = x[:,-del_num:,:]
        generator.drop_temp()
        outputs = torch.zeros(y.size(0), y.size(1),2)
        
        loss_node = []
        dif_node = []
        for j in range(HYP['node_num']-del_num):
            # zero grad
            op_net.zero_grad()
            op_dyn.zero_grad()
            opt_states.zero_grad()
            
            x_un_pre = states_learner(states_id.cuda())# .detach() 
            x_hypo = torch.cat((x_kn,x_un_pre.float()),1)
    
            hypo_adj = generator.sample_all()
            hypo_adj[:-del_num,:-del_num] = observed_adj

            adj_col = hypo_adj[:,j].cuda()# hard = true
        
            num = int(HYP['node_num'] / HYP['node_size'])
            remainder = int(HYP['node_num'] % HYP['node_size'])
            if remainder == 0:
                num = num - 1
                
            ex,y_hat = dyn_isom(x_hypo, adj_col, j, num, HYP['node_size'])
            loss = loss_fn(y_hat,y[:,j])
            # print(loss.item())
                    
            # backward and optimize
            loss.backward()
            # cut gradient in case nan shows up
            U.clip_grad_norm_(generator.gen_matrix, 0.000075)

            #op_net.step()
            op_dyn.step()
            op_net.step()
            opt_states.step()

            # use outputs to caculate mse
            outputs[:, j, :] = y_hat

            # record
            loss_node.append(loss.item())

        # dif_batch.append(np.mean(dif_node))
        loss_batch.append(np.mean(loss_node))
        
        #s_mae = accu(x_un,x_un_pre)
        # statesmae_batch.append(s_mae)
        # print("idx",s_mae)
        
    # used for more than 10 nodes
    # op_net.zero_grad()
    # loss = torch.abs(torch.sum(generator.sample_all())- infer_edges) * HYP['lr_stru']  # 极大似然
    # loss.backward()
    # op_net.step()

    return torch.mean(torch.FloatTensor(loss_batch))

def onehot_state(x_un_pre):
    if x_un_pre.shape[1]==1:
        state = torch.argmax(x_un_pre,2)
        pre_state = torch.cat((state,1-state),1).unsqueeze(1)
    else:
        state = torch.argmax(x_un_pre,2).unsqueeze(2)
        pre_state = torch.cat((state,1-state),2)  
    return pre_state

def cal_states_loss():
    mae =[]
    for data,states_id in zip(train_loader,train_index):
        x = data[0].float().to(device)
        x_un = x[:, -del_num:,:]
        x_un_pre = states_learner(states_id.cuda()).detach()
        pre_state = onehot_state(x_un_pre)
        state_erro_rate = abs(torch.argmax(x_un,2)-torch.argmax(pre_state,2)).float().mean()
        state_erro_rate = state_erro_rate.cpu().numpy()
        mae.append(state_erro_rate) 
        
    return np.mean(mae)


loss_all = []
for j in range(HYP["epoch_num"]):
    loss = train_dyn_gen_state_voter()
    print(j,loss)
    (index_order,auc_net,precision,kn_un_precision,un_un_precision) = part_constructor_evaluator_sgm(generator,1,object_matrix,HYP["node_num"],del_num)
    loss_all.append(loss)
    x_state_erro = cal_states_loss()
    print(x_state_erro)