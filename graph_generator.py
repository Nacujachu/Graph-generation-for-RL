import torch 
import torch.nn as nn
import networkx as nx
import numpy as np
from collections import namedtuple
import random
import math
import warnings
from GCN import GCN_network
warnings.filterwarnings("ignore")


class graph_generation_GCPN(nn.Module):
    


class graph_generation_network(nn.Module):

    '''
    Input: one graph laplacian
    Output: type,density
    '''

    def __init__(self , device = 'cuda:0'):
        super().__init__()
        
        self.graph_gcn = GCN_network(in_feature=16 , out_feature=32 , n_hidden_layer=2 , device = device)
        self.lstm = nn.LSTM(32 , 32).type(torch.DoubleTensor)
        self.type_layer = nn.Linear(32 , 2)
        self.density_layer = nn.Linear(32 , 1)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self , graph_lap , hidden):
        
        #hidden = self.init_hidden()
        graph_embedding = self.graph_gcn(graph_lap)
        graph_embedding = graph_embedding.view(1,1,-1)
        
        out , hidden = self.lstm(graph_embedding , hidden)
        

        graph_type = self.softmax(self.type_layer(out).view(-1) )

        out , hidden = self.lstm(graph_embedding , hidden)

        graph_density = self.sigmoid(self.density_layer(out).view(-1))

        return graph_type , graph_density
        
        
    def init_hidden(self):
        return ((torch.zeros(1, 1, 32).to(self.device),torch.zeros(1, 1, 32).to(self.device) ))


class graph_generation_network_MLP(nn.Module):
    def __init__(self , device = 'cuda:0'):
        super().__init__()
        
        self.graph_gcn = GCN_network(in_feature=16 , out_feature=32 , n_hidden_layer=2 , device = device)
        self.density_layer = nn.Linear(32 , 1)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def forward(self , graph_lap ):
        
        #hidden = self.init_hidden()
        graph_embedding = self.graph_gcn(graph_lap)
        graph_embedding = graph_embedding.view(1,-1)
        graph_density = self.density_layer(graph_embedding)
        
        return self.sigmoid(graph_density) 
        



class graph_agent(nn.Module):
    def __init__(self , in_feature=16 , out_feature=32 , n_hidden_layer=2 , device = 'cuda:0' ):
        
        self.network = graph_generation_network_MLP(in_feature=in_feature , out_feature=out_feature , n_hidden_layer=n_hidden_layer , device = device )
        self.states = []
        self.actions = []
        self.rewards = []
        self.hidden = self.init_hidden()
    def clear_transitions(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def forward(self , graph_lap , hidden):
        
        return 


    def init_hidden(self):
        return ((torch.zeros(1, 1, 32).to(self.device),torch.zeros(1, 1, 32).to(self.device) ))
