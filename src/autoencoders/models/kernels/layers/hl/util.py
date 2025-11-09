import torch

def random_reduce(err):
    weights = torch.rand_like(err)
    loss = weights * (err) ** 2
    return loss.mean()