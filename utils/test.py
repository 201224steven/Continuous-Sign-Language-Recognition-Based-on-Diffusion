import torch
import torch.nn as nn
import numpy as np

a = torch.randn((2,  3, 4))
b = torch.randn((2, 5, 4))
print(a.shape)
print(b.transpose(1, 2).shape)
m = torch.matmul(a, b.transpose(1, 2))
print(m.shape)