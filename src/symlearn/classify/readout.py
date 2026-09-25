import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Readout(nn.Module):
    """
    Attention block intended to aggregate the decisions of pre-Readout models.
    """

    def __init__(
            self, 
            input_dim: int = 224, 
            hidden_dim: int = 32,
            preR_dim: int = 32,
            num_hidden: int = 1,
            num_classes: int = 4, 
            num_heads: int = 1, 
            num_preR: int = 3, 
            dropout: float = 0.0,) -> None:
        
        super(Readout, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.preR_dim = preR_dim
        self.num_hidden = num_hidden
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_preR = num_preR
        self.dropout = dropout
        
        if self.num_heads > 1:
            self.multi_head = True
        else:
            self.multi_head = False
        
        self.batch_norm = nn.BatchNorm1d(self.preR_dim*self.num_preR, affine=False)  
        self.fc_embeds = nn.Linear(self.preR_dim*self.num_preR, self.hidden_dim)
        nn.init.kaiming_normal_(self.fc_embeds.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc_embeds.bias)

        if self.num_hidden == 1:
            layer = nn.Linear(self.num_classes*self.num_preR + self.hidden_dim, self.hidden_dim)
            nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
            nn.init.zeros_(layer.bias)
        
            self.linears = nn.ModuleList([layer])
            
        else:
            first_layer = nn.Linear(self.num_classes*self.num_preR + self.hidden_dim, self.hidden_dim)
            nn.init.kaiming_normal_(first_layer.weight, nonlinearity='leaky_relu')
            nn.init.zeros_(first_layer.bias)
            
            self.linears = nn.ModuleList([first_layer])

            for _ in range(self.num_hidden-1):
                hidden_layer = nn.Linear(self.hidden_dim, self.hidden_dim)
                nn.init.kaiming_normal_(hidden_layer.weight, nonlinearity='leaky_relu')
                nn.init.zeros_(hidden_layer.bias)
            
                self.linears.append(hidden_layer)

        self.out = nn.Linear(self.hidden_dim, self.num_classes)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

        if self.multi_head:
            self.attn_embed = nn.Linear(self.num_classes*self.num_preR, self.hidden_dim*self.num_heads)
            
            self.multihead_attn = nn.MultiheadAttention(
                self.num_heads*self.hidden_dim, 
                self.num_heads, 
                dropout=self.dropout, 
                batch_first=True
                )
            
            self.attn_out = nn.Linear(self.num_heads*self.hidden_dim, self.num_classes*self.num_preR)
        else:
            self.attn_embed = nn.Linear(self.num_classes*self.num_preR, self.hidden_dim)
            self.attn_out = nn.Linear(self.hidden_dim, self.num_classes*self.num_preR)
            
    def input(self, 
              ind_embeds: Tensor) -> Tensor:
        
        embed = self.batch_norm(ind_embeds)
        embed = self.fc_embeds(embed)
        return embed
        
    def forward(self, 
                logits: Tensor, 
                ind_embeds: Tensor) -> Tensor:
        
        logits = self.attn_embed(logits)
        q = logits
        k = logits
        v = logits

        if not self.multi_head:
            logits = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
        else:
            logits, _ = self.multihead_attn(q, k, v, need_weights=False)

        logits = F.tanh(self.attn_out(logits))
        embed = F.tanh(self.input(ind_embeds))
        
        x = torch.cat([logits,embed],dim=1)

        for layer in self.linears:
            x = layer(x)
            x = F.leaky_relu(x)
        
        return self.out(x)