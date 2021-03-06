import torch 
import torch.nn as nn
import networkx as nx
import numpy as np
from collections import namedtuple
from torch.autograd import Variable
from GCN import GCN_network
import random
import math
import warnings
from utils import validation 
from MVC_env import MVC_environement
from DQN_network import embedding_network , to_sparse_tensor
warnings.filterwarnings("ignore")


'''
neighbors_sum = torch.sparse.mm(adj_list , emb_matrix[0])
neighbors_sum = neighbors_sum.view(batch_size , neighbors_sum.shape[0] , neighbors_sum.shape[1])
'''





experience = namedtuple("experience" , ['graph','Xv','action','reward','next_Xv','is_done'])
fitted_q_exp = namedtuple("fitted_exp" , ['graph','Xv','action','reward'])

class replay_buffer():
    def __init__(self , max_size):
        self.buffer = np.zeros(  [max_size],dtype = experience)
        self.max_size = max_size
        self.size = 0
        self.idx = -1
    def push(self , new_exp):
        if(self.size >= self.max_size):
            self.idx = (self.idx+1) % self.max_size
        else:
            self.idx = self.idx + 1
            self.size += 1
        
        self.buffer[self.idx] = new_exp
    
    def sample(self , batch_size):
        batch = np.random.choice(np.arange(self.size) , size = batch_size , replace=False)
        
        return self.buffer[[batch]]

    def clear_buffer(self):
        self.size = 0
        self.idx = -1
class Agent(nn.Module):
    def __init__(self , emb_dim = 64 , T = 5,device = 'cuda:0' , init_factor = 10 , w_scale = 0.01 , init_method = 'normal' , 
    replay_size = 500000 , PG = False , global_net = False , fitted_Q = True , fix_seed = True , lr = 1e-4 , adaption_test = False , weight_decay = 0.0):
        

        super().__init__()
        if fix_seed:
            torch.cuda.manual_seed_all(19960214)
            torch.manual_seed(19960214)
            np.random.seed(19960214)
            random.seed(19960214)
        if global_net:
            self.dqn =\
            embedding_network_conditional(emb_dim = emb_dim , T = T,device = device , init_factor = init_factor , w_scale = w_scale , init_method = init_method,fix_seed = fix_seed).double().to(device)
            self.target_net = \
            embedding_network_conditional(emb_dim = emb_dim , T = T,device = device , init_factor = init_factor , w_scale = w_scale , init_method = init_method,fix_seed = fix_seed ).double().to(device)
        else:
            self.dqn =\
            embedding_network(emb_dim = emb_dim , T = T,device = device , init_factor = init_factor , w_scale = w_scale , init_method = init_method, fix_seed = fix_seed).double().to(device)
            self.target_net = \
            embedding_network(emb_dim = emb_dim , T = T,device = device , init_factor = init_factor , w_scale = w_scale , init_method = init_method , fix_seed = fix_seed).double().to(device)

        self.target_net.load_state_dict(self.dqn.state_dict())

        self.buffer = replay_buffer(replay_size)
        ##
        if adaption_test:
            enc_parameters_list = list(self.dqn.W1.parameters()) + list(self.dqn.W2.parameters()) + list(self.dqn.W3.parameters()) + list(self.dqn.W4.parameters())
            dec_parameters_list =  list(self.dqn.W6.parameters()) +  list(self.dqn.W5.parameters()) + list(self.dqn.W7.parameters()) 
            self.enc_optimizer = torch.optim.Adam( enc_parameters_list , lr = lr , amsgrad=False , weight_decay = weight_decay)
            self.dec_optimizer = torch.optim.Adam( dec_parameters_list , lr = lr , amsgrad=False , weight_decay = weight_decay)
            
            self.optimizer = torch.optim.Adam( parameters_list , lr = lr , amsgrad=False , weight_decay = weight_decay)
        else:
            self.optimizer = torch.optim.Adam(self.dqn.parameters() , lr = lr , weight_decay = weight_decay)

        self.loss_func = torch.nn.MSELoss()

        self.device = device

        self.steps_done = 0

        self.EPS_END = 0.05
        self.EPS_START = 1.0
        self.EPS_DECAY = 20000
        self.N = 0
        self.N_STEP = 2
        self.PG = PG
        self.non_selected = []
        self.selected = []
        self.fitted_Q = fitted_Q
        self.episode_done = 0
        if PG:
            self.log_probs = []
            self.rewards = []
            self.actions = []
            self.dones = []
            self.softmax = torch.nn.Softmax(dim = 1)
    def forward(self , graph , Xv ):
        
        if self.device != None:
            graph = graph.to(self.device)
            Xv = Xv.to(self.device)
        
        if self.PG:
            select_index = (Xv[0].long() == 1).view(-1)
            out = self.dqn(graph , Xv)
            out[0][select_index] = -float('inf')
            return self.softmax(out)
        else:
            return self.dqn(graph , Xv)

    def new_epsiode(self):
        self.N = 0
        self.selected = []

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def reset(self):
        self.clear_buffer()
        self.steps_done = 0
        self.episode_done = 0

    def take_action(self , graph , Xv , is_validation = False):
        if self.PG:
            val = self.forward(graph , Xv)[0]
            m = torch.distributions.categorical.Categorical(probs = val)
            action = m.sample()
            if(is_validation == False):
                self.actions.append(action)
                self.log_probs.append(m.log_prob(action))
                
            else:
                self.selected.append(action)
            return action.item()
        else:
            if(len(self.selected) == 0):
                self.non_selected = [i for i in range(graph.shape[1])]

            eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
            rand_val = np.random.uniform()
            if is_validation:
                rand_val = 999
            #print(is_validation , rand_val)
            if  rand_val > eps_threshold  :
                select_index = (Xv[0].long() == 1).view(-1)
                val = self.forward(graph , Xv)[0]
                val[self.selected] = -float('inf')
                #print(val)
                action = int(torch.argmax(val).item())
                self.selected.append(action)
                self.non_selected.remove(action)
            else:
                #print(self.non_selected)
                action = int(np.random.choice(self.non_selected))
                self.non_selected.remove(action)
                self.selected.append(action)
            if(is_validation == False):
                self.steps_done = self.steps_done + 1
            return action
    def get_val_result(self, validation_graph):

        objective_vals = []
        for g in validation_graph:
            env = MVC_environement(g)
            Xv , graph = env.reset_env()
            graph = torch.unsqueeze(graph,  0)
            Xv = Xv.clone()
            Xv = Xv.cuda()
            graph = graph.to(self.device)
            done = False
            self.non_selected = list(np.arange(env.num_nodes))
            self.selected = []
            while done == False:
                #Xv = Xv.cuda()
                Xv = Xv.to(self.device)
                val = self.forward(graph , Xv)[0]
                #val[selected] = -float('inf')
                #print(val)
                #action = int(torch.argmax(val).item())
                action = self.take_action(graph , Xv , is_validation = True)
                
                Xv_next , reward , done = env.take_action(action)
                #non_selected.remove(action)
                #selected.append(action)
                Xv = Xv_next
            #print(selected)
            objective_vals.append(len(self.selected))
        return sum(objective_vals)/len(objective_vals)
    
    def get_val_result_batch(self , validation_graph , return_list = False):
        N = len(validation_graph)
        all_graphs = []
        all_Xv = []
        all_envs = []
        for g in validation_graph:
            env = MVC_environement(g)
            all_envs.append(env)
            Xv , graph = env.reset_env()
            graph = torch.unsqueeze(graph,  0)
            all_graphs.append(graph)
            all_Xv.append(Xv)
        all_graphs = torch.cat(all_graphs , 0).to(self.device)
        all_Xv = torch.cat(all_Xv , 0).to(self.device)
        all_selected = [[] for _ in range(N)]
        all_dones = [False for _ in range(N)]
        all_done = False
        done_count = 0
        while all_done == False:
            q_val = self.dqn(all_graphs , all_Xv )
            for i in range(N):

                if all_dones[i]:
                    continue

                q_val[i][ all_selected[i] ] = -float('inf')
                action = torch.argmax(q_val[i]).item()
                all_selected[i].append(action)
                _,_,done = all_envs[i].take_action(action)
                all_Xv[i][action] = 1
                if done:
                    all_dones[i] = True
                    done_count += 1
            if(done_count == N):
                break
            #print(all_Xv)
            #print(q_val.shape)
            #del all_graphs
            #del all_Xv
            #break
        objective_vals = []
        for s in all_selected:
            objective_vals.append(len(s))
        
        if return_list:
            return objective_vals

        return sum(objective_vals)/len(objective_vals)
        #del all_graphs,all_Xv

    def store_transition(self , new_exp):
        self.buffer.push(new_exp)
    
    

    def train(self , batch_size = 64 , fitted_Q = False) :
        if self.PG:
            R = 0
            rewards_list = []
            for r , d in zip(self.rewards , self.dones):
                R = r + R
                rewards_list.insert(0 , R)
                if(d):
                    R = 0
            #print(rewards_list)
            rewards = torch.FloatTensor(rewards_list)
            rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
            probs = torch.Tensor(self.log_probs).to(self.device)
            rewards = rewards.to(self.device)
            
            loss = 0

            for lgp , r in zip(self.log_probs , rewards):
                loss += -lgp*r

            # print(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.rewards = [] 
            self.dones = []
            self.log_probs = []
        else:
            if(self.buffer.size < batch_size):
                return 
            
            batch = self.buffer.sample(batch_size)
            batch = experience(*zip(*batch))
            batch_graph = torch.cat(batch.graph)
            batch_state = torch.cat(batch.Xv)
            batch_action = torch.cat(batch.action)
            batch_reward = torch.cat(batch.reward).double()
            batch_next_state = torch.cat(batch.next_Xv)
            non_final_mask = torch.tensor(tuple(map(lambda s : s is not True, batch.is_done)),dtype = torch.uint8)

            non_final_graph = batch_graph[non_final_mask]
            non_final_next_state = batch_next_state[non_final_mask]

            next_state_value = torch.zeros(batch_size).detach().double()
            if self.device == None:
                self.device = 'cuda:0'
            device = self.device
            batch_graph = batch_graph.to(device)
            batch_state = batch_state.to(device)
            batch_action = batch_action.to(device)
            batch_reward = batch_reward.to(device)
            batch_next_state = batch_next_state.to(device)
            next_state_value = next_state_value.to(device)
            non_final_graph = non_final_graph.to(device)
            non_final_next_state = non_final_next_state.to(device)

            pred_q = self.dqn(batch_graph , batch_state ).gather(1 , batch_action.view(-1,1)).view(-1)

            if self.fitted_Q:
                next_state_value[non_final_mask] = self.dqn(non_final_graph , non_final_next_state).max(1)[0].detach()
                #self.buffer.clear_buffer()
            else:
                next_state_value[non_final_mask] = self.target_net(non_final_graph , non_final_next_state).max(1)[0].detach()

            expected_q = next_state_value + batch_reward
            loss = self.loss_func(pred_q , expected_q)

            #print(loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



    def train_with_graph(self , g ):

        '''
        g: networkx graph
        '''
        N_STEP = 2
        env = MVC_environement(g)
        Xv , graph = env.reset_env()
        Xv = Xv.clone()
        graph = torch.unsqueeze(graph,  0)
        done = False
        fitted_experience_list = []
        reward_list = []
        #self.new_epsiode()
        self.non_selected = list(np.arange(env.num_nodes))
        self.selected = []
        self.N = 0
        while done == False:
            
            action = self.take_action(graph , Xv)
            Xv_next , reward , done = env.take_action(action)
            Xv_next = Xv_next.clone()
            fit_ex = fitted_q_exp(graph , Xv , action , reward)
            fitted_experience_list.append(fit_ex)
            self.N += 1 
            reward_list.append(reward)
            if self.N >= N_STEP:
                n_reward = sum(reward_list)
                n_prev_ex = fitted_experience_list[0]
                n_graph = n_prev_ex.graph
                n_Xv = n_prev_ex.Xv
                n_action = n_prev_ex.action
                ex = experience(n_graph , n_Xv , torch.tensor([n_action]) , torch.tensor([n_reward]) , Xv_next , done)
                self.store_transition(ex)
                fitted_experience_list.pop(0)
                reward_list.pop(0)
            Xv = Xv_next
            self.train()
        self.episode_done += 1
        if  self.episode_done > 0 and self.episode_done %8 == 0:
            #print(self.steps_done)
            self.update_target_network()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.dqn.state_dict())

        
    def save_weights(self , file_name):
        torch.save(self.dqn.state_dict() , file_name)

    def load_weights(self , file_name = None ,state_dict = None):
        
        if file_name == None and state_dict == None:
            print('no state to load')

        if state_dict is not None:
            self.dqn.load_state_dict(state_dict)
        else:
            self.dqn.load_state_dict(torch.load(file_name))