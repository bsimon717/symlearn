import numpy as np
import torch
from torch._higher_order_ops import map
import torch.nn as nn
import torch.nn.functional as F

def embed_sim(x1, x2):
    cosine_sim = F.cosine_similarity(x1, x2, dim=0)
    return (cosine_sim+1)/2

def embed_summand(src, aux, delta=0.5):
    return torch.exp( (1/delta)*embed_sim(src, aux) ) - 1

def embed_loss(src, auxs):
    num_aux = len(auxs)
    temp_func = lambda aux: embed_summand(src, aux)
    temp_func = torch.vmap(temp_func)

    auxs = torch.stack(auxs)
    sum_terms = temp_func(auxs)
    sum_terms = torch.sum(sum_terms, dim=0)
    pre_factor = 1/num_aux
    
    L_embed = pre_factor*sum_terms

    return L_embed.mean()