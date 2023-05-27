import torch
import torch.nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch.nn import Sequential, Linear, ReLU


###################    
##### TM-HGNN #####   
###################

class TM_HGNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):  
        super(TM_HGNN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_mask, batch):
        # All hyperedges
        x = self.conv1(x, edge_index)
        x = x.relu()
        # Note hyperedges
        idx_1 = torch.where(edge_mask==1)[0]
        x = self.conv2(x, edge_index[:, idx_1])   
        x = x.relu()
        # Taxonomy hyperedges
        idx_2 = torch.where(edge_mask==2)[0]
        x = self.conv3(x, edge_index[:, idx_2])        

        # Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # final classifier
        x = F.dropout(x, p=0.3, training=self.training)
        x = self.lin(x)
        
        return x
    

if __name__ == '__main__':
    print("TM-HGNN")