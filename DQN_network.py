import torch 
import torch.nn as nn
import networkx as nx
import numpy as np

'''
neighbors_sum = torch.sparse.mm(adj_list , emb_matrix[0])
neighbors_sum = neighbors_sum.view(batch_size , neighbors_sum.shape[0] , neighbors_sum.shape[1])
'''


def to_sparse_tensor(x):
    if len(x.shape) == 3:
        indices = torch.nonzero(x).t()
        values = x[indices[0], indices[1],indices[2]]
        sparse1 = torch.sparse.FloatTensor(indices, values, x.size())
        return sparse1
    elif len(x.shape) == 2:
        indices = torch.nonzero(x).t()
        values = x[indices[0], indices[1]]
        sparse1 = torch.sparse.FloatTensor(indices, values, x.size())
        return sparse1


class embedding_network(nn.Module):
    
    def __init__(self , emb_dim = 64 , T = 4,device = None , init_factor = 10 , w_scale = 0.01 , init_method = 'normal'):
        super().__init__()
        self.emb_dim = emb_dim
        self.T = T
        self.W1 = nn.Linear( 1 , emb_dim , bias = False)
        self.W2 = nn.Linear(emb_dim , emb_dim , bias = False)
        self.W3 = nn.Linear(emb_dim , emb_dim , bias = False)
        self.W4 = nn.Linear( 1 , emb_dim , bias = False)
        self.W5 = nn.Linear(emb_dim*2,1 , bias = False)
        self.W6 = nn.Linear(emb_dim , emb_dim , bias = False)
        self.W7 = nn.Linear(emb_dim , emb_dim , bias = False)
        
        std = 1/np.sqrt(emb_dim)/init_factor
        
        for W in [self.W1 , self.W2 , self.W3 , self.W4 , self.W5 , self.W6 , self.W7]:
            if init_method == 'normal':
                nn.init.normal_(W.weight , 0.0 , w_scale)
            else:
                nn.init.uniform_(W.weight , -std , std)
        ##nn.init.uniform_(self.W1.weight , -std, std )
        #nn.init.uniform_(self.W2.weight, -std, std )
        #nn.init.uniform_(self.W3.weight, -std, std )
        #nn.init.uniform_(self.W4.weight, -std, std )
        #nn.init.uniform_(self.W5.weight, -std, std )
        #nn.init.uniform_(self.W6.weight, -std, std )
        #nn.init.uniform_(self.W7.weight, -std, std )
        self.device = device
        self.relu = nn.ReLU()
        
    def forward(self , graph , Xv ):
        device = self.device
        batch_size = Xv.shape[0]
        n_vertex = Xv.shape[1]
        graph_edge = torch.unsqueeze(graph , 3)
        

        emb_matrix = torch.zeros([batch_size , n_vertex , self.emb_dim ]).type(torch.DoubleTensor)
        
        if 'cuda' in Xv.type():
            if device == None:
                emb_matrix = emb_matrix.cuda()
            else:
                emb_matrix = emb_matrix.cuda(device)
        for t in range(self.T):
            neighbor_sum = torch.bmm(graph , emb_matrix )
            v1 = self.W1(Xv)
            v2 = self.W2(neighbor_sum)
            v3 = self.W4(graph_edge)
            v3 = self.W3(torch.sum(v3 , 2))
            
            v = v1 + v2 + v3
            #emb_matrix = v.clone()
            emb_matrix = self.relu(v)
            #print(v1 , v2 , v3)
            #print('=================')
            #print(v[0][0])
        #print(emb_matrix.shape)
        emb_sum = torch.sum(emb_matrix , 1)
        v6 = self.W6(emb_sum)
        v6 = v6.repeat(1,n_vertex )
        v6 = v6.view(batch_size , n_vertex , self.emb_dim)
        v7 = self.W7(emb_matrix)
        ct = self.relu(torch.cat([v6 , v7] , 2))
        
        return torch.squeeze(self.W5(ct) , 2)