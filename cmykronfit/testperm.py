import numpy as np
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
sigma = np.eye(5)
sigma[0][1] = 1
print(sigma)
n1 = 1
n2 = 2
ran = np.arange(5)
sigma_later=SwapElement(sigma,n1,n2)
ran[n1],ran[n2] = ran[n2],ran[n1]
print(sigma_later,ran)
p = np.eye(5)[ran]
sigma_before = np.dot(np.dot(p,sigma_later),p.T)
print(sigma_before)