import torch
import torch.nn as nn
import torch.nn.functional as F

class Readout(nn.Module):
    def __init__(self, input_dim=224, hidden_dim=32, num_classes=4, dropout=0.0, num_heads=1, num_preR=3, preR_dim=32):
        super(Readout, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_preR = num_preR
        self.preR_dim = preR_dim
        self.dropout = 0

        self.num_heads = num_heads
        if self.num_heads > 1:
            self.multi_head = True
        else:
            self.multi_head = False
        
        self.batch_norm = nn.BatchNorm1d(self.preR_dim*self.num_preR, affine=False)  
        self.fc0 = nn.Linear(self.preR_dim*self.num_preR, self.hidden_dim)
        nn.init.kaiming_normal_(self.fc0.weight, nonlinearity='linear')
        nn.init.zeros_(self.fc0.bias)
        
        self.fc1 = nn.Linear(self.num_classes*self.num_preR + self.hidden_dim, self.hidden_dim*2)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.zeros_(self.fc1.bias)
        
        self.fc2 = nn.Linear(self.hidden_dim*2, self.num_classes)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        if self.multi_head:
            self.attn_embed = nn.Linear(self.num_classes*self.num_preR, self.hidden_dim*self.num_heads)
            self.multihead_attn = nn.MultiheadAttention(self.num_heads*self.hidden_dim, self.num_heads, dropout=self.dropout, batch_first=True)
            self.attn_out = nn.Linear(self.num_heads*self.hidden_dim, self.num_classes*self.num_preR)
        else:
            self.attn_embed = nn.Linear(self.num_classes*self.num_preR, self.hidden_dim)
            self.attn_out = nn.Linear(self.hidden_dim, self.num_classes*self.num_preR)
            
    def input(self, ind_embeds):
        embed = self.batch_norm(ind_embeds)
        embed = self.fc0(embed)
        return embed
        
    def forward(self, logits, ind_embeds):
        x = self.attn_embed(logits)
        q = x
        k = x
        v = x

        if not self.multi_head:
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout)
            x = F.tanh(self.attn_out(x))
            embed = F.tanh(self.input(ind_embeds))

            x = self.fc1(torch.cat([x,embed],dim=1))
            x = F.leaky_relu(x)
            
            return self.fc2(x)
            
        else:
            x, _ = self.multihead_attn(q, k, v, need_weights=False)
            x = F.tanh(self.attn_out(x))
            embed = F.tanh(self.input(ind_embeds))

            x = self.fc1(torch.cat([x,embed],dim=1))
            x = F.leaky_relu(x)
            
            return self.fc2(x)