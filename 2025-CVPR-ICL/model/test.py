import torch

a = torch.zeros((4,4))
b = torch.ones((4,4))
c = torch.stack([a,b],dim=0).max(dim=0)

print(c)