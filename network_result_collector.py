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
from DQN_network import Agent
warnings.filterwarnings("ignore")



class Trainer():
    def __init__(self):
        
        self.agent = Agent()
        

    def train(self , g , num_eps = 20):

        N_STEP = 2
        fitted_q_exp = namedtuple("fitted_exp" , ['graph','Xv','action','reward'])
        experience = namedtuple("experience" , ['graph','Xv','action','reward','next_Xv','is_done'])
        EPS_START = 1.00
        EPS_END = 0.05
        EPS_DECAY = 500
        steps_done = 0
        for e in range(num_eps):

            env = MVC_environement(g)
            Xv , graph = env.reset_env()
            Xv = Xv.clone()
            graph = torch.unsqueeze(graph,  0)
            done = False
            non_selected = list(np.arange(env.num_nodes))
            selected = []
            N = 0
            fitted_experience_list = []
            reward_list = []
            self.agent.new_epsiode()
            while done == False:
                eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
                if np.random.uniform() > eps_threshold:
                    val = self.agent(graph , Xv)[0]
                    val[selected] = -float('inf')
                    action = int(torch.argmax(val).item())
                else:
                    action = int(np.random.choice(non_selected))
                Xv_next , reward , done = env.take_action(action)
                Xv_next = Xv_next.clone()
                fit_ex = fitted_q_exp(graph , Xv , action , reward)
                fitted_experience_list.append(fit_ex)
                non_selected.remove(action)
                selected.append(action)
                N += 1 
                reward_list.append(reward)
                
                if N >= N_STEP:
                    n_reward = sum(reward_list)
                    n_prev_ex = fitted_experience_list[0]
                    n_graph = n_prev_ex.graph
                    n_Xv = n_prev_ex.Xv
                    n_action = n_prev_ex.action
                    ex = experience(n_graph , n_Xv , torch.tensor([n_action]) , torch.tensor([n_reward]) , Xv_next , done)
                    self.agent.store_transition(ex)
                    fitted_experience_list.pop(0)
                    reward_list.pop(0)
                    
                Xv = Xv_next
                steps_done += 1
                self.agent.train(batch_size = 8 , fitted_Q = True)
            #print(eps_threshold)

    def train_and_report_improvement(self, train_g , val_g ):
        self.save_weights()
        before_train = self.agent.get_val_result(val_g)
        self.train(train_g , num_eps = 10)
        after_train = self.agent.get_val_result(val_g)
        self.load_weights()
        return before_train , after_train

    def save_weights(self):
        self.state_dict = self.agent.dqn.state_dict()
    def load_weights(self):
        self.agent.load_weights(state_dict = self.state_dict)