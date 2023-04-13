import numpy as np
import torch

data = np.load("./data/00000e74ad.npy", mmap_mode='r')
datatorch = torch.from_numpy(data)
print(data)
print(datatorch.shape)