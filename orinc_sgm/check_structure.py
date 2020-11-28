from tools_changedelmethod import *
import matplotlib.pyplot as plt 
adj_address = '/data/chenmy/voter/seed2051email14320000-adjmat.pickle'
series_address = '/data/chenmy/voter/seed2051email14320000-series.pickle'
train_loader, val_loader, test_loader, object_matrix = load_bn_ggn(series_address,adj_address,batch_size= 1024,seed = 153)

for data in train_loader:
    x = data[0]
    x_kn = x[:,0,0]
    plt.figure()
    plt.plot(x_kn)    
    plt.show()
    break




