from DQN_network import embedding_network
import torch 
import torch.nn as nn
import networkx as nx
import numpy as np
from collections import namedtuple
import random
import math
import warnings
from utils import validation 
from MVC_env import MVC_environement
from utils import mvc_bb
warnings.filterwarnings("ignore")





class gnn_predict_network(nn.Module):

    def __init__(self , emb_dim = 64 , T = 4 , device = 'cuda:0' , init_factor = 10 , w_scale = 0.01 , init_method = 'normal' ):
        super().__init__()
        self.gnn =\
            embedding_network(emb_dim = emb_dim , T = T,device = device , init_factor = init_factor , w_scale = w_scale , init_method = init_method).double().to(device)
        
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self , graph , Xv):

        x = self.gnn(graph,Xv)

        return self.sigmoid(x)



class training_data_generator():
    
    def __init__(self , original_graph_distribution = ('er' , 0.15) , T = 5 , K = 100 , N = 50 , train_epoch = 50 , num_hint = 5, device = 'cuda:0'):
        '''
        T iterations
        K graph per iteration
        N graph size
        '''
        
    
        self.num_hint = num_hint
        self.train_epoch = train_epoch
        self.T = T
        self.K = K
        self.N = N
        self.graph_list = []
        self.ground_truth = []
        graph_type , p = original_graph_distribution
        for _ in range(K):
            if graph_type == 'er':
                g = nx.erdos_renyi_graph(n = N , p = p)
            else:
                raise BaseException("No good")
            self.graph_list.append(g)
            #self.ground_truth.append(mvc_bb(g))
        
        self.population = self.graph_list[:]
        self.gnn_net = gnn_predict_network(device = device).to(device)



    def network_train(self):
        


        
        cur_graphs = self.population[:]
        cur_target = []
        training_target = []
        training_graphs = []
        training_feature = []
        for g in cur_graphs:
            sol = mvc_bb(g)
            env = MVC_environement(g)
            Xv , graph = env.reset_env()
            idx = np.random.choice(sol , self.num_hint )
            graph = torch.unsqueeze(graph , 0)
            training_graphs.append(graph)
            Xv[0,[idx],0] = 1
            training_feature.append(Xv)
            ans = torch.zeros_like(Xv).squeeze(2)
            ans[0,sol] = 1
            training_target.append(ans)

        training_graphs = torch.cat(training_graphs , 0)
        training_target = torch.cat(training_target , 0)
        training_feature = torch.cat(training_feature , 0)
        print(training_graphs.shape , training_target.shape, training_feature.shape)

