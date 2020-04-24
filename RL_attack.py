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
from DQN_network import Agent

warnings.filterwarnings("ignore")


class attack_rl():

    def __init__(self,original_graph_distribution = ('er' , 0.15) , T = 5 , K = 100 , 
    N = 50 , train_epoch = 10 , device = 'cuda:1' , time_report = False , num_modify = 150 , train_on_whole = False , fix_seed = True , uniform_init = False,
    weight_decay = 0.0001 , prioritize_rp = False):

        if fix_seed:
            print('seed fixed')
            torch.cuda.manual_seed_all(19960214)
            torch.manual_seed(19960214)
            np.random.seed(19960214)
            random.seed(19960214)

        self.rl_agent = Agent(fix_seed = True,device = device ,fitted_Q = False , replay_size = 500000 , weight_decay = weight_decay , prioritize_rp = prioritize_rp).to(device)
        self.train_on_whole = train_on_whole
        self.graph_list = []
        self.T = T
        self.K = K
        self.N = N
        self.train_epoch = train_epoch
        graph_type , p = original_graph_distribution
        self.graph_list = []
        self.offspring = []
        self.ground_truth = []
        self.device = device
        self.loss_history = []
        self.val_history = []
        self.time_report = time_report
        for i in range(K):
            if graph_type == 'er':
                if uniform_init:
                    p = random.uniform(0.1,0.5)
                g = nx.erdos_renyi_graph(n = N , p = p)
            
            elif graph_type == 'mix':
                if i % 2 == 0:
                    g = nx.erdos_renyi_graph(n = N , p = p)
                else:
                    g = nx.barabasi_albert_graph(n = N , m = 4)

            else:
                raise BaseException("No good")
            self.graph_list.append(g)
            #self.ground_truth.append(mvc_bb(g))
        self.num_modify = num_modify
        self.population = self.graph_list[:]


    def run_GA(self , validation_graphs = None):
        self.network_train()
        for T in range(self.T):
            print("T ",T)
            if self.time_report:
                s = timer()
            #self.crossover()
            self.simple_generation()
            if self.time_report:
                e = timer()
                print('crossover ',e-s)
            if self.time_report:
                s = timer()
            self.selection()
            if self.time_report:
                e = timer()
                print('selection ',e-s)
            
            #self.mutation()
            if self.time_report:
                s = timer()
            if validation_graphs is not None:
                self.get_val_result_batch(validation_graphs)
                if self.time_report:
                    e = timer()
                print('validation ',e-s)
            if self.time_report:
                s = timer()
            self.network_train()
            if self.time_report:
                e = timer()
                print('training ',e-s)
    
    def get_val_result_batch(self , validation_graphs):
        
        res = []
        for vg in validation_graphs:
            res.append(self.rl_agent.get_val_result_batch(vg))
        self.val_history.append(res)
    

    def simple_generation(self):
        self.offspring = []
        for _ in range(2*self.K):
            p = random.uniform(0,1)
            self.offspring.append(nx.erdos_renyi_graph(n = self.N , p = p ) )
        

    def network_train(self):
        
        #self.rl_agent.reset()

        if self.train_on_whole:
            train_graphs = self.graph_list[:]
        else:
            train_graphs = self.population[:]

        for e in range(self.train_epoch):
            for g in train_graphs:
                self.rl_agent.train_with_graph(g)


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
        for _ in range(self.K*2):
            g1 , g2 = random.sample( self.population , 2   )
            new_g = self.crossover_two_graph(g1 , g2)
            self.offspring.append(new_g)



    
    def mutation_graph(self,  g , num_modify = 150):
        
        modify_edges = random.randint( 0 , self.num_modify)

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

        population_targets = []
        all_population = self.population + self.offspring
        for g in all_population:
            population_targets.append(len(mvc_bb(g)))


        
        population_out = self.rl_agent.get_val_result_batch(all_population , return_list = True)
        approx_val = []
        #print(population_out)
        #print(population_targets)
        for opt , out in zip(population_targets , population_out):
            if opt > 0:
                approx_val.append(-(out / opt))
            else:
                approx_val.append(0)
        print(approx_val)
        population_idx = np.argsort(approx_val)
        new_pop = []
        for pidx in population_idx[:self.K]:
            #print(approx_val[pidx])
            new_pop.append(all_population[pidx])

        #print(population_loss)
        self.graph_list = self.graph_list + new_pop
        self.population = new_pop[:]

    def save_weight(self , fname = 'rl_attack_model/tmp.pkl'):
        if 'rl_attack_model/' not in fname:
            raise BaseException("need to store in right dir")
        self.rl_agent.save_weights(fname)
        #torch.save(self.gnn_net.state_dict() , fname)