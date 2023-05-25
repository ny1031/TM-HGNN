import torch
import torch.nn
from torch.nn import Linear, Embedding
import torch.nn.functional as F
torch.manual_seed(42)

import numpy as np


class Handle_data(object):
    def __init__(self, type, tokenizer):
        self.type = type
        self.tokenizer = tokenizer

    def tokenize(self, data):
        data.edge_mask = data.edge_index_mask
        del data.edge_index_mask            
        # type
        data.x_type = data.x_n[:, 0].to(torch.long)
        # index for word node
        data.x_idx = data.x_n[:, 1].to(torch.long)
        # x
        data.x = data.x_n[:, 4:104].to(torch.float32)

        ### embed hyperedge ###
            
        # TAXONOMY Hyperedge
        t_idx = (data.x_type==2.0).nonzero().squeeze()
        t_emb = Embedding(15,100)
        t_emb.weight.requires_grad=False
        tax_emb = t_emb(torch.index_select(data.batch_t, 0, t_idx)).to(torch.float32)
        data.x[t_idx.long(),:] = tax_emb

        # NOTE Hyperedge
        # Requires Note id & Taxonomy id
        n_idx = (data.x_type==1.0).nonzero().squeeze()
        n_emb = Embedding(100,100)   
        n_emb.weight.requires_grad=False
        note_emb = n_emb(torch.index_select(data.batch_n, 0, n_idx)).to(torch.float32)
        # get hour_n 
        dic = {0.0:0, 24.0:1, 48.0:2}
        h_emb = Embedding(3,100)
        h_emb.weight.requires_grad=False

        hour_emb = h_emb(torch.tensor(list(map(lambda x : dic[x], data.hour_n.squeeze(-1).numpy()))))

        # replace with sum embedding
        data.x[n_idx.long(),:] = note_emb + hour_emb 
            
        data.y = data.y_p.to(torch.float32)


        return data

    def cut_off(self, data):

        return data

    def __call__(self, data):

        data = self.tokenize(data)
        data = self.cut_off(data)


        return data