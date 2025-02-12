import numpy as np

x= np.load('celeba_60x50.npy')
print(x.shape)
print(x[0])
x_mean = np.mean(x,axis=0)
x-x_mean