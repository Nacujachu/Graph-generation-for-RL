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
import torch.utils.data as Data
from copy import deepcopy
import numpy as np
from timeit import default_timer as timer
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
    
    def __init__(self , original_graph_distribution = ('er' , 0.15) , T = 5 , K = 100 , N = 50 , train_epoch = 10 , num_hint = 10, device = 'cuda:0' , time_report = False):
        '''
        T iterations
        K graph per iteration
        N graph size
        '''
        
        self.time_report = time_report
        self.num_hint = num_hint
        self.train_epoch = train_epoch
        self.T = T
        self.K = K
        self.N = N
        self.graph_list = []
        self.offspring = []
        self.ground_truth = []
        self.device = device
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

    def run_GA(self):
        self.network_train()
        for T in range(self.T):
            print("T ",T)
            if self.time_report:
                s = timer()
            self.crossover()
            if self.time_report:
                e = timer()
                print('crossover ',e-s)
            if self.time_report:
                s = timer()
            self.selection()
            if self.time_report:
                e = timer()
                print('selection ',e-s)
            
            self.mutation()

            if self.time_report:
                s = timer()
            self.network_train()
            if self.time_report:
                e = timer()
                print('training ',e-s)

    def crossover_two_graph(self , g1 , g2 , pr = 0.5):

        g3 = nx.trivial_graph() 
        g3.add_nodes_from(g1.nodes())
        g3.add_edges_from(g1.edges() & g2.edges()  )
        
        diff_g1_g2 = g1.edges() - g2.edges()
        diff_g2_g1 = g2.edges() - g2.edges()
        g3.add_edges_from( random.sample(  diff_g1_g2  , int(len(diff_g1_g2)*pr ) ) )
        g3.add_edges_from( random.sample(  diff_g2_g1  , int(len(diff_g2_g1)*pr ) ) )
        
        return g3

    def crossover(self):
        self.offspring = []
        for _ in range(self.K):
            g1 , g2 = random.sample( self.population , 2   )
            new_g = self.crossover_two_graph(g1 , g2)
            self.offspring.append(new_g)



    
    def mutation_graph(self,  g , num_modify = 150):
        
        modify_edges = random.randint(-num_modify*.5 , num_modify)

        nodes = [ i for i in range(len(g.nodes())) ]

        while(len(g.edges()) and modify_edges < 0):
            e = random.sample(g.edges() ,1 )[0]
            v , u = e
            g.remove_edge(v , u)
            modify_edges += 1


        while(modify_edges > 0):
            u , v = random.sample(nodes , k = 2)
            if (u,v) in g.edges():
                continue
            else:
                g.add_edge(u , v)
                modify_edges -= 1
        '''
        for _ in range(num_modify):
            e = random.sample(g.edges() ,1 )[0]
            v , u = e
            c = random.sample(g.nodes() , 1)[0]
            if(u != c):
                g.remove_edge(v , u)
                g.add_edge(v , c)
        '''


    def mutation(self):
        
        for g in self.population:
            self.mutation_graph(g)




    def selection(self):
        population_loss = []
        offspring_loss = []
        loss_fn = torch.nn.BCELoss()
        population_targets = []
        population_graphs = []
        population_features = []
        all_population = self.population + self.offspring
        for g in all_population:
            graph , Xv , ans = self.get_G_X_target(g)
            population_graphs.append(graph)
            population_features.append(Xv)
            population_targets.append(ans)

        population_graphs = torch.cat(population_graphs , 0).to(self.device)
        population_targets = torch.cat(population_targets , 0).to(self.device)
        population_features = torch.cat(population_features , 0).to(self.device)
        
        population_out = self.gnn_net(population_graphs , population_features)
        for out , target in zip(population_out , population_targets):
            #print( out.shape, target.shape)
            population_loss.append(-loss_fn(out , target).item())

        population_idx = np.argsort(population_loss)
        new_pop = []
        for pidx in population_idx[:self.K]:
            #print(population_loss[pidx])
            new_pop.append(all_population[pidx])

        #print(population_loss)
        self.graph_list = self.graph_list + new_pop
        self.population = new_pop[:]
        

    
    def get_G_X_target(self , g):
        sol = mvc_bb(g)
        env = MVC_environement(g)
        Xv , graph = env.reset_env()
        if len(sol) >= self.num_hint:
            idx = np.random.choice(sol , self.num_hint )
        else:
            idx = sol
        graph = torch.unsqueeze(graph , 0)
        Xv[0,[idx],0] = 1
        ans = torch.zeros_like(Xv).squeeze(2)
        ans[0,sol] = 1

        return graph , Xv , ans
    def network_train(self):
        
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.gnn_net.parameters() , lr = 2e-3)

        
        cur_graphs = self.population[:]
        cur_target = []
        training_targets = []
        training_graphs = []
        training_features = []
        for g in cur_graphs:
            
            graph , Xv , ans = self.get_G_X_target(g)

            training_graphs.append(graph)
            training_features.append(Xv)
            training_targets.append(ans)

        training_graphs = torch.cat(training_graphs , 0)
        training_targets = torch.cat(training_targets , 0)
        training_features = torch.cat(training_features , 0)
        #print(training_graphs.shape , training_targets.shape, training_features.shape)

        dtset = Data.TensorDataset(training_graphs , training_features , training_targets)
        loader = Data.DataLoader(dataset = dtset , batch_size = 16 , shuffle = True)
        losses_list = []
        for e in range(self.train_epoch):
            loss_sum = 0
            for graphs , Xvs , y in loader:
                #print(graphs.shape , Xvs.shape , y.shape)
                graphs = graphs.to(self.device)
                Xvs = Xvs.to(self.device)
                y = y.to(self.device)
                out = self.gnn_net(graphs,Xvs)
                loss = loss_fn(out , y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            losses_list.append(loss_sum)

        
        return losses_list