import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init
import numpy as np

use_cuda = torch.cuda.is_available()


class GGN_IO(nn.Module):
    def __init__(self, node_num, dim, hid):
        super(GGN_IO, self).__init__()
        self.node_num = node_num
        self.dim = dim
        self.hid = hid
        self.n2e = nn.Linear(2 * dim, hid)
        self.e2e = nn.Linear(hid, hid)
        self.e2n = nn.Linear(hid, hid)
        self.n2e = nn.Linear(2 * dim, hid)
        self.output = nn.Linear(dim + hid, dim)

    def forward(self, x, adj_col, i):
        # x : features of all nodes at time t,[n*d]
        # adj_col : i th column of adj mat,[n*1]
        # i : just i
        starter = x
        ender = x[i].repeat(self.node_num).view(self.node_num, -1)
        x = torch.cat((starter, ender), 1)
        # cat（）拼接矩阵函数，第二个参数是拼接的维度。（0：横着拼，1：按列拼）
        x = F.relu(self.n2e(x))
        x = F.relu(self.e2e(x))
        x = x * adj_col.unsqueeze(1).expand(self.node_num, self.hid)
        x = torch.sum(x, 0)
        x = self.e2n(x)
        x = torch.cat((starter[i], x), dim=-1)
        x = self.output(x)

        # skip connection
        x = starter[i] + x
        return x


class GGN_IO_B(nn.Module):
    """docstring for GGN_IO_B"""

    def __init__(self, node_num, dim, hid):
        super(GGN_IO_B, self).__init__()
        self.node_num = node_num
        self.dim = dim
        self.hid = hid
        self.n2e = nn.Linear(2 * dim, hid)
        self.e2e = nn.Linear(hid, hid)
        self.e2n = nn.Linear(hid, hid)
        self.n2n = nn.Linear(hid, hid)
        self.output = nn.Linear(dim + hid, dim)

    def forward(self, x, adj_col, i):
        # print(adj_col)
        # d()
        # x : features of all nodes at time t,[b*n*d]
        # adj_col : i th column of adj mat,[n*1]
        # i : just i
        starter = x  # 128,10,4
        ender = x[:, i, :]  # 128,4
        ender = ender.unsqueeze(1)  # 128,1,4

        ender = ender.expand(starter.size(0), starter.size(1), starter.size(2))  # 128,10,4
        x = torch.cat((starter, ender), 2)  # 128,10,8
        x = F.relu(self.n2e(x))  # 128,10,256

        x = F.relu(self.e2e(x))  # 128,10,256
        x = x * adj_col.unsqueeze(1).expand(self.node_num, self.hid)  # 128,10,256

        x = torch.sum(x, 1)  # 128,256
        x = F.relu(self.e2n(x))  # 128,256
        x = F.relu(self.n2n(x))  # 128,256

        x = torch.cat((starter[:, i, :], x), dim=-1)  # 128,256+4
        x = self.output(x)  # 128,4

        # skip connection
        # x = starter[:,i,:]+x # dont want in CML
        return x



class IO_B(nn.Module):
    """docstring for IO_B"""

    def __init__(self, dim, hid):
        super(IO_B, self).__init__()
        self.dim = dim
        self.hid = hid
        self.n2e = nn.Linear(2 * dim, hid)
        self.e2e = nn.Linear(hid, hid)
        self.e2n = nn.Linear(hid, hid)
        self.n2n = nn.Linear(hid, hid)
        self.output = nn.Linear(dim + hid, dim)
        # self.logsoftmax = nn.LogSoftmax(dim=1)

    # def forward(self, x, adj_col, i):
    #     # print(adj_col)
    #     # d()
    #     # x : features of all nodes at time t,[b*n*d]
    #     # adj_col : i th column of adj mat,[n*1]
    #     # i : just i
    #     starter = x  # 128,10,4
    #     ender = x[:, i, :]  # 128,4
    #     ender = ender.unsqueeze(1)  # 128,1,4
    #
    #     ender = ender.expand(starter.size(0), starter.size(1), starter.size(2))  # 128,10,4
    #     x = torch.cat((starter, ender), 2)  # 128,10,8
    #     # print(x.size())
    #     x = F.relu(self.n2e(x))  # 128,10,256
    #
    #     x = F.relu(self.e2e(x))  # 128,10,256
    #     x = x * adj_col.unsqueeze(1).expand(adj_col.size(0), self.hid)  # 128,10,256
    #
    #     x = torch.sum(x, 1)  # 128,256
    #     x = F.relu(self.e2n(x))  # 128,256
    #     x = F.relu(self.n2n(x))  # 128,256
    #
    #     x = torch.cat((starter[:, i, :], x), dim=-1)  # 128,256+4
    #     x = self.output(x)  # 128,4
    #
    #     # skip connection
    #     # x = starter[:,i,:]+x # dont want in CML
    #     return x

    def forward(self, x, adj_col, i, num, node_size):
        # print(adj_col)
        # d()
        # x : features of all nodes at time t,[b*n*d]
        # adj_col : i th column of adj mat,[n*1]
        # i : just i
        #print(i)
        starter = x[:, i, :]
        #print(starter.shape)
        
        x_total_sum = 0
        for n in range(num + 1):
            if n != num:
                current_x = x[:, n * node_size:(n + 1) * node_size, :]
                current_adj_col = adj_col[n * node_size:(n + 1) * node_size]
            else:
                current_x = x[:, n * node_size:, :]
                current_adj_col = adj_col[n * node_size:]
            ender = x[:, i, :]  # 128,4
            # 增加一维
            ender = ender.unsqueeze(1)  # 128,1,4
            # 扩展某一维 ender尺寸变为128,10,4
            ender = ender.expand(current_x.size(0), current_x.size(1), current_x.size(2))  # 128,10,4
            # 拼接stater和ender
            c_x = torch.cat((current_x, ender), 2)  # 128,10,8
            c_x = F.relu(self.n2e(c_x))  # 128,10,256

            c_x = F.relu(self.e2e(c_x))  # 128,10,256
            c_x = c_x * current_adj_col.unsqueeze(1).expand(current_adj_col.size(0), self.hid)  # 128,10,256

            current_x_sum = torch.sum(c_x, 1)  # 128,256

            x_total_sum = x_total_sum + current_x_sum

        x = F.relu(self.e2n(x_total_sum))  # 128,256
        x = F.relu(self.n2n(x))  # 128,256

        x = torch.cat((starter, x), dim=-1)  # 128,256+4
        x = self.output(x)  # 128,4
        # x = self.logsoftmax(x)

        # skip connection
        # x = starter[:,i,:]+x # dont want in CML
        return x

    # def forward1(self, x, adj_col,ender):
    #     # print(adj_col)
    #     # d()
    #     # x : features of all nodes at time t,[b*n*d]
    #     # adj_col : i th column of adj mat,[n*1]
    #     # i : just i
    #     starter = x  # 128,10,4
    #     # 128,4
    #     # 增加一维
    #     ender = ender.unsqueeze(1)  # 128,1,4
    #     # 扩展某一维 ender尺寸变为128,10,4
    #     ender = ender.expand(starter.size(0), starter.size(1), starter.size(2))  # 128,10,4
    #     # 拼接stater和ender
    #     x = torch.cat((starter, ender), 2)  # 128,10,8
    #     x = F.relu(self.n2e(x))  # 128,10,256
    #
    #     x = F.relu(self.e2e(x))  # 128,10,256
    #     x = x * adj_col.unsqueeze(1).expand(adj_col.size(0), self.hid)  # 128,10,256
    #     x = torch.sum(x, 1)  # 128,256
    #     return x
    #
    # def forward2(self, x,starter):
    #     x = F.relu(self.e2n(x))  # 128,256
    #     x = F.relu(self.n2n(x))  # 128,256
    #     x = torch.cat((starter, x), dim=-1)  # 128,256+4
    #     x = self.output(x)  # 128,4
    #
    #     # skip connection
    #     # x = starter[:,i,:]+x # dont want in CML
    #     return x

    
    
class IO_B_discrete(nn.Module):
    """docstring for IO_B"""

    def __init__(self, dim, hid):
        super(IO_B_discrete, self).__init__()
        self.dim = dim
        self.hid = hid
        self.n2e = nn.Linear(2 * dim, hid)
        self.e2e = nn.Linear(hid, hid)
        self.e2n = nn.Linear(hid, hid)
        self.n2n = nn.Linear(hid, hid)
        self.output = nn.Linear(dim + hid, dim)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, adj_col, i, num, node_size):
        starter = x[:, i, :]
        
        x_total_sum = 0
        for n in range(num + 1):
            if n != num:
                current_x = x[:, n * node_size:(n + 1) * node_size, :]
                current_adj_col = adj_col[n * node_size:(n + 1) * node_size]
            else:
                current_x = x[:, n * node_size:, :]
                current_adj_col = adj_col[n * node_size:]
            ender = x[:, i, :]  # 128,4

            ender = ender.unsqueeze(1)  # 128,1,4

            ender = ender.expand(current_x.size(0), current_x.size(1), current_x.size(2))  # 128,10,4
            
            # 拼接stater和ender
            c_x = torch.cat((current_x, ender), 2)  # 128,10,8
            c_x = F.relu(self.n2e(c_x))  # 128,10,256

            
            c_x = F.relu(self.e2e(c_x))  # 128,10,256
            c_x = c_x * current_adj_col.unsqueeze(1).expand(current_adj_col.size(0), self.hid)  # 128,10,256

            current_x_sum = torch.sum(c_x, 1)  # 128,256

            x_total_sum = x_total_sum + current_x_sum

        x = F.relu(self.e2n(x_total_sum))  # 128,256
        x = F.relu(self.n2n(x))  # 128,256

        x = torch.cat((starter, x), dim=-1)  # 128,256+4
        ex = self.output(x)  # 128,4
        x = self.logsoftmax(ex)
        return x


class GGN_SEP_B(nn.Module):
    """分开自己和邻居"""

    def __init__(self, node_num, dim, hid):
        super(GGN_SEP_B, self).__init__()
        self.node_num = node_num
        self.dim = dim
        self.hid = hid
        self.n2e = nn.Linear(2 * dim, hid)
        self.e2e = nn.Linear(hid, hid)
        self.e2n = nn.Linear(hid, hid)
        self.n2n = nn.Linear(hid, hid)
        self.selflin = nn.Linear(dim, hid)
        self.output = nn.Linear(2 * hid, dim)

    def forward(self, x, adj_col, i):
        # print(adj_col)
        # d()
        # x : features of all nodes at time t,[b*n*d]
        # adj_col : i th column of adj mat,[n*1]
        # i : just i
        starter = x  # 128,10,4
        ender = x[:, i, :]  # 128,4
        # print('ender')
        # print(ender.size())
        ender = ender.unsqueeze(1)  # 128,1,4
        # print('ender unsqueeze')
        # print(ender.size())

        ender = ender.expand(starter.size(0), starter.size(1), starter.size(2))  # 128,10,4
        # print('ender expand')
        # print(ender.size())

        x = torch.cat((starter, ender), 2)  # 128,10,8
        # print('cat')
        # print(x.size())

        x = F.relu(self.n2e(x))  # 128,10,256
        # print('n2e')
        # print(x.size())

        x = F.relu(self.e2e(x))  # 128,10,256
        # print('e2e')
        # print(x.size())
        x = x * adj_col.unsqueeze(1).expand(self.node_num, self.hid)  # 128,10,256
        # print('times the col of adj mat')
        # print(x.size())

        x = torch.sum(x, 1)
        # print('reduced sum')
        # print(x.size())
        x = F.relu(self.e2n(x))
        # print('e2n')
        # print(x.size())
        x = F.relu(self.n2n(x))
        # print('n2n')
        # print(x.size())

        # self information transformation
        starter = F.relu(self.selflin(starter[:, i, :]))
        # print('linear transformation for self node')
        # print(starter.size())

        # cat them together
        x = torch.cat((starter, x), dim=-1)
        # print('cat self and neighbor')
        # print(x.size())
        x = self.output(x)
        # print('output')
        # print(x.size())
        # d()

        # skip connection
        # x = starter[:,i,:]+x # dont want in CML
        return x


# 此类为一个利用Gumbel softmax生成离散网络的类
class Gumbel_Generator(nn.Module):
    def __init__(self, sz=10, temp=1, temp_drop_frac=0.9999):
        super(Gumbel_Generator, self).__init__()
        self.sz = sz
        self.tau = temp
        self.drop_fra = temp_drop_frac
        self.gen_matrix = Parameter(torch.rand(sz, sz, 2))
        self.temperature = temp
        self.temp_drop_frac = temp_drop_frac

    def sample_adj_i(self, i, hard=True, sample_time=1):
        # mat = torch.zeros(sample_time,self.sz)
        # for m in range(sample_time):
        # 	mat[m] = F.gumbel_softmax(self.gen_matrix[:,i], tau=self.tau, hard=hard)[:,0]
        # res = torch.sum(mat,0) / sample_time
        # return res
        return F.gumbel_softmax(self.gen_matrix[:, i] + 1e-8, tau=self.tau, hard=hard)[:, 0]

    def ana_one_para(self):
        print(self.gen_matrix[0][0])
        return 1

    def sample_all(self, hard=True, sample_time=1):
        adj = torch.zeros(self.sz, self.sz)
        for i in range(adj.size(0)):
            temp = self.sample_adj_i(i, hard=hard, sample_time=sample_time)
            adj[:, i] = temp
        return adj

    def drop_temp(self):
        self.tau = self.tau * self.drop_fra


#############
# Functions #
#############
def gumbel_sample(shape, eps=1e-20):
    u = torch.rand(shape)
    gumbel = - np.log(- np.log(u + eps) + eps)
    if use_cuda:
        gumbel = gumbel.cuda()
    return gumbel


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    # gumbel_sample 返回一个sample采样
    y = logits + gumbel_sample(logits.size())
    return torch.nn.functional.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
    logits: [batch_size, n_class] unnormalized log-probs
    temperature: non-negative scalar
    hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
    [batch_size, n_class] sample from the Gumbel-Softmax distribution.
    If hard=True, then the returned sample will be one-hot, otherwise it will
    be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = logits.size()[-1]
        y_hard = torch.max(y.data, 1)[1]
        y = y_hard
    return y


def get_offdiag(sz):
    ## 返回一个大小为sz的下对角线矩阵
    offdiag = torch.ones(sz, sz)
    for i in range(sz):
        offdiag[i, i] = 0
    if use_cuda:
        offdiag = offdiag.cuda()
    return offdiag


#####################
# Network Generator #
#####################


'''在网络补全任务中生成未知部分结构'''
class Gumbel_Generator_nc(nn.Module):
    def __init__(self, sz=10,del_num = 1,temp=10, temp_drop_frac=0.9999):
        super(Gumbel_Generator_nc, self).__init__()
        self.sz = sz
        self.del_num = del_num
        self.gen_matrix = Parameter(torch.rand(del_num*(2*sz-del_num-1)//2, 2)) #cmy get only unknown part parameter
        self.temperature = temp
        self.temp_drop_frac = temp_drop_frac

    def drop_temp(self):
        # 降温过程
        self.temperature = self.temperature * self.temp_drop_frac

    def sample_all(self, hard=False):
        self.logp = self.gen_matrix
        if use_cuda:
            self.logp = self.gen_matrix.cuda()
        
        out = gumbel_softmax(self.logp, self.temperature, hard)
        if hard:
            hh = torch.zeros((self.del_num*(2*self.sz - self.del_num-1)//2,2))
            for i in range(out.size()[0]):
                hh[i, out[i]] = 1
            out = hh
                             
        out = out[:, 0]

        if use_cuda:
            out = out.cuda()
        
        matrix = torch.zeros(self.sz,self.sz).cuda()
        left_mask = torch.ones(self.sz,self.sz)
        left_mask[:-self.del_num,:-self.del_num] = 0
        left_mask = left_mask - torch.diag(torch.diag(left_mask))
        un_index = torch.triu(left_mask).nonzero()
        matrix[(un_index[:,0],un_index[:,1])] = out
        out_matrix = matrix + matrix.T
        # out_matrix = out[:, 0].view(self.gen_matrix.size()[0], self.gen_matrix.size()[0])
        return out_matrix
     
    def init(self, mean, var):
        init.normal_(self.gen_matrix, mean=mean, std=var)
        
'''生成连续节点未知部分状态'''
class Generator_states(nn.Module):
    def __init__(self,dat_num,del_num):
        super(Generator_states, self).__init__()
        self.embeddings = nn.Embedding(dat_num, del_num)
    def forward(self, idx):
        pos_probs = torch.sigmoid(self.embeddings(idx)).unsqueeze(2)
        return pos_probs

'''生成离散节点未知部分的状态'''
class Generator_states_discrete(nn.Module):
    def __init__(self,dat_num,del_num):
        super(Generator_states_discrete, self).__init__()
        self.embeddings = nn.Embedding(dat_num, del_num)
    def forward(self, idx):
        pos_probs = torch.sigmoid(self.embeddings(idx)).unsqueeze(2)
        probs = torch.cat([pos_probs, 1 - pos_probs], 2)
        return probs
    
# 非对称的NC
'''在网络补全任务中生成未知部分结构'''
class Gumbel_Generator_nc_asy(nn.Module):
    def __init__(self, sz=10,del_num = 1,temp=10, temp_drop_frac=0.9999):
        super(Gumbel_Generator_nc_asy, self).__init__()
        self.sz = sz
        self.del_num = del_num
        self.gen_matrix = Parameter(torch.rand(del_num*(2*sz-del_num-1), 2)) #cmy get only unknown part parameter
        self.temperature = temp
        self.temp_drop_frac = temp_drop_frac

    def drop_temp(self):
        # 降温过程
        self.temperature = self.temperature * self.temp_drop_frac

    def sample_all(self, hard=False):
        self.logp = self.gen_matrix
        if use_cuda:
            self.logp = self.gen_matrix.cuda()
        
        out = gumbel_softmax(self.logp, self.temperature, hard)
        if hard:
            hh = torch.zeros((self.del_num*(2*self.sz - self.del_num-1),2))
            for i in range(out.size()[0]):
                hh[i, out[i]] = 1
            out = hh
                             
        out = out[:, 0]

        if use_cuda:
            out = out.cuda()
        
        matrix = torch.zeros(self.sz,self.sz).cuda()
        left_mask = torch.ones(self.sz,self.sz)
        left_mask[:-self.del_num,:-self.del_num] = 0
        left_mask = left_mask - torch.diag(torch.diag(left_mask))
        un_index = left_mask.nonzero()
        matrix[(un_index[:,0],un_index[:,1])] = out
        out_matrix = matrix
        # out_matrix = out[:, 0].view(self.gen_matrix.size()[0], self.gen_matrix.size()[0])
        return out_matrix
     
    def init(self, mean, var):
        init.normal_(self.gen_matrix, mean=mean, std=var)


    
# 此类为一个利用Gumbel softmax生成离散网络的类
class Gumbel_Generator_Old(nn.Module):
    def __init__(self, sz=10, temp=10, temp_drop_frac=0.9999):
        super(Gumbel_Generator_Old, self).__init__()  # 将类Gumbe_Generator_Old 对象转换为类 nn.Module 的对象
        self.sz = sz
        # 将一个不可训练的类型Tensor 转换成可以训练的类型Parameter 经过这个类型转换这个self，***就成了模型的中一部分
        # 成为了模型中根据训练可以改动的参数了
        self.gen_matrix = Parameter(torch.rand(sz, sz, 2))  # torch.rand()返回一个张量，包含了从区间（0，1）随机抽取的一组随机数
        self.new_matrix = Parameter(torch.zeros(5, 5, 2))
        # gen_matrix 为邻接矩阵的概率
        self.temperature = temp
        self.temp_drop_frac = temp_drop_frac

    def symmetry(self):
        matrix = self.gen_matrix.permute(2, 0, 1)
        temp_matrix = torch.triu(matrix, 1) + torch.triu(matrix, 1).permute(0, 2, 1)
        self.gen_matrix.data = temp_matrix.permute(1, 2, 0)

    def drop_temp(self):
        # 降温过程
        self.temperature = self.temperature * self.temp_drop_frac

    def sample_all(self, hard=False, epoch=1):
        # 采样——得到一个邻接矩阵
        # self.symmetry()
        self.logp = self.gen_matrix.view(-1, 2)  # view先变成一维的tensor，再按照参数转换成对应维度的tensor
        out = gumbel_softmax(self.logp, self.temperature, hard)
        if hard:
            hh = torch.zeros(self.gen_matrix.size()[0] ** 2, 2)
            for i in range(out.size()[0]):
                hh[i, out[i]] = 1
            out = hh
        if use_cuda:
            out = out.cuda()
        out_matrix = out[:, 0].view(self.gen_matrix.size()[0], self.gen_matrix.size()[0])
        return out_matrix

    def sample_small(self, list, hard=False):
        indices = np.ix_(list, list)
        self.logp = self.gen_matrix[indices].view(-1, 2)  # view先变成一维的tensor，再按照参数转换成对应维度的tensor

        out = gumbel_softmax(self.logp, self.temperature, hard)

        # hard 干什么用的
        if hard:
            hh = torch.zeros(self.gen_matrix[indices].size()[0] ** 2, 2)
            for i in range(out.size()[0]):
                hh[i, out[i]] = 1
            out = hh
        if use_cuda:
            out = out.cuda()
        out_matrix = out[:, 0].view(len(list), len(list))
        return out_matrix

    def sample_adj_ij(self, list, j, hard=False, sample_time=1):
        # self.logp = self.gen_matrix[:,i]
        self.logp = self.gen_matrix[list, j]

        out = gumbel_softmax(self.logp, self.temperature, hard=hard)
        if use_cuda:
            out = out.cuda()
        # print(out)
        if hard:
            out_matrix = out.float()
        else:
            out_matrix = out[:, 0]
        return out_matrix

    def sample_adj_i(self, i, hard=False, sample_time=1):
        # self.symmetry()
        self.logp = self.gen_matrix[:, i]
        out = gumbel_softmax(self.logp, self.temperature, hard=hard)
        if use_cuda:
            out = out.cuda()
        # print(out)
        if hard:
            out_matrix = out.float()
        else:
            out_matrix = out[:, 0]
        return out_matrix

    def get_temperature(self):
        return self.temperature

    def get_cross_entropy(self, obj_matrix):
        # 计算与目标矩阵的距离
        logps = F.softmax(self.gen_matrix, 2)
        logps = torch.log(logps[:, :, 0] + 1e-10) * obj_matrix + torch.log(logps[:, :, 1] + 1e-10) * (1 - obj_matrix)
        result = - torch.sum(logps)
        result = result.cpu() if use_cuda else result
        return result.data.numpy()

    def get_entropy(self):
        logps = F.softmax(self.gen_matrix, 2)
        result = torch.mean(torch.sum(logps * torch.log(logps + 1e-10), 1))
        result = result.cpu() if use_cuda else result
        return (- result.data.numpy())

    def randomization(self, fraction):
        # 将gen_matrix重新随机初始化，fraction为重置比特的比例
        sz = self.gen_matrix.size()[0]
        numbers = int(fraction * sz * sz)
        original = self.gen_matrix.cpu().data.numpy()

        for i in range(numbers):
            ii = np.random.choice(range(sz), (2, 1))
            z = torch.rand(2).cuda() if use_cuda else torch.rand(2)
            self.gen_matrix.data[ii[0], ii[1], :] = z

    def init(self, mean, var):
        init.normal_(self.gen_matrix, mean=mean, std=var)

