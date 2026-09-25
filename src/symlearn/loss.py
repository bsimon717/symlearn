import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

def embed_sim(x1: Tensor, x2: Tensor) -> Tensor:
    """
    Computes the cosine similarity between two embeddings scaled to the range [0,1].
    """
    
    cosine_sim = F.cosine_similarity(x1, x2, dim=0)
    return (cosine_sim+1)/2

def embed_summand(src: Tensor, aux: Tensor, delta: float = 0.5) -> Tensor:
    """
    Computes a sum-term in a pre-Readout model's Embedding Loss.
    """
    return torch.exp( (1/delta)*embed_sim(src, aux) ) - 1

def embed_loss(src: Tensor, auxs: Tensor) -> Tensor:
    """
    Computes a pre-Readout model's Embedding Loss.
    """

    num_aux = len(auxs)
    temp_func = lambda aux: embed_summand(src, aux)
    temp_func = torch.vmap(temp_func)

    auxs = torch.stack(auxs)
    sum_terms = temp_func(auxs)
    sum_terms = torch.sum(sum_terms, dim=0)
    pre_factor = 1/num_aux
    
    L_embed = pre_factor*sum_terms

    return L_embed.mean()