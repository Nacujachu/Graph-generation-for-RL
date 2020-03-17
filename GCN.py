import torch
import torch.nn as nn
import torch.functional as F
from torch.nn.parameter import Parameter
import pickle
def pickle_save(data , file_name):
    with open(file_name , 'wb') as f:
        pickle.dump(data , f)
def pickle_load( file_name):
    with open(file_name , 'rb') as f:
        return pickle.load(f)

import math

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


class GCN_layer(nn.Module):
    def __init__(self , in_feature , out_feature ,device = 'cuda:0'):
        super(GCN_layer,self).__init__()
        self.theta1 = Parameter(torch.FloatTensor(in_feature,out_feature)).type(torch.DoubleTensor).to(device)
        self.theta2 = Parameter(torch.FloatTensor(in_feature,out_feature)).type(torch.DoubleTensor).to(device)

        std = 1./math.sqrt(self.theta1.size(1))
        self.theta1.data.uniform_(-std,std)
        self.theta2.data.uniform_(-std,std)

    def forward(self , lap , input_feature):
        out1 = torch.matmul(input_feature , self.theta1)
        
        if 'sparse' in lap.type():
            out2= torch.sparse.mm(lap , input_feature)
            #print(out2.type())
            #out2 = out2.cpu().to_dense().cuda()
            out2= torch.matmul(out2 , self.theta2)
        else:
            #print( lap.type() , input_feature.type())
            out2 = torch.matmul(lap , input_feature) 
            out2 = torch.matmul(out2,self.theta2)
        return out1 + out2



class GCN_network(nn.Module):
    def __init__(self , in_feature , out_feature , n_hidden_layer , device = 'cuda:0' ):
        super().__init__()
        self.gcn_in = GCN_layer(in_feature,out_feature , device = device)
        self.relu = nn.ReLU()
        hidden_gcn = [GCN_layer(out_feature,out_feature,device) for i in range(n_hidden_layer)]
        self.gcns = nn.Sequential(*hidden_gcn)
        self.gcn_out = GCN_layer(out_feature , out_feature,device)
        self.in_feature = in_feature
        self.device = device

    def forward(self , graph_lap):
        
        batch_size = graph_lap.shape[0]
        node_num = graph_lap.shape[1]
        X = torch.ones(batch_size , node_num , self.in_feature).type(torch.DoubleTensor).to(self.device)
        
        x = self.gcn_in(graph_lap , X)
        x = self.relu(x)
        for layer in self.gcns:
            
            x = layer(graph_lap , x)
            x = self.relu(x)
        out = self.gcn_out(graph_lap,x)
        out = torch.mean(out , dim = 1)
        #print(out.shape)
        return out

