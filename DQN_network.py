import torch 
import torch.nn as nn
import networkx as nx
import numpy as np
from collections import namedtuple
from torch.autograd import Variable
from GCN import GCN_network
import torch.nn.functional as F
import random
import math
import warnings
from utils import validation 
from MVC_env import MVC_environement
from prioritize_replay import PrioritizedReplayBuffer , ReplayBuffer
from schedules import LinearSchedule
from copy import deepcopy
warnings.filterwarnings("ignore")

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


class embedding_network_conditional(nn.Module):
    def __init__(self , emb_dim = 64 , T = 4,device = None , init_factor = 10 , w_scale = 0.01 , init_method = 'normal'):
        super().__init__()
        self.emb_dim = emb_dim
        self.T = T
        self.W1 = nn.Linear( 1 , emb_dim , bias = False)
        self.W2 = nn.Linear(emb_dim , emb_dim , bias = False)
        self.W3 = nn.Linear(emb_dim , emb_dim , bias = False)
        self.W4 = nn.Linear( 1 , emb_dim , bias = False)
        self.W5 = nn.Linear(emb_dim*3,1 , bias = False)
        self.W6 = nn.Linear(emb_dim , emb_dim , bias = False)
        self.W7 = nn.Linear(emb_dim , emb_dim , bias = False)
        self.global_net = GCN_network(in_feature = 16 , out_feature = emb_dim , n_hidden_layer = 2 , device = device)
        std = 1/np.sqrt(emb_dim)/init_factor
        
        for W in [self.W1 , self.W2 , self.W3 , self.W4 , self.W5 , self.W6 , self.W7]:
            if init_method == 'normal':
                nn.init.normal_(W.weight , 0.0 , w_scale)
            else:
                nn.init.uniform_(W.weight , -std , std)
        self.device = device
        self.relu = nn.ReLU()
        
    def forward(self , graph , Xv ):
        


        if len(graph.size()) == 2:
            graph = torch.unsqueeze(graph,  0)
    
        device = self.device
        batch_size = Xv.shape[0]
        n_vertex = Xv.shape[1]
        graph_edge = torch.unsqueeze(graph , 3)
        
        global_embedding = self.global_net(graph)
        #print(global_embedding.shape)
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
        #print(emb_sum.shape)
        
        v6 = self.W6(emb_sum)
        v6 = v6.repeat(1,n_vertex )
        v6 = v6.view(batch_size , n_vertex , self.emb_dim)

        glob = global_embedding.repeat(1 , n_vertex)
        glob = glob.view(batch_size , n_vertex , self.emb_dim)

        v7 = self.W7(emb_matrix)
        ct = self.relu(torch.cat([v6 , v7 , glob] , 2))
        #print(ct.shape)
        return torch.squeeze(self.W5(ct) , 2)

class embedding_network(nn.Module):
    
    def __init__(self , emb_dim = 64 , T = 4,device = None , init_factor = 10 , w_scale = 0.01 , init_method = 'normal' , fix_seed = True):
        if fix_seed:
            torch.cuda.manual_seed_all(19960214)
            torch.manual_seed(19960214)
            np.random.seed(19960214)
            random.seed(19960214)

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
        self.device = device
        self.relu = nn.ReLU(inplace = True)
        
    def forward(self , graph , Xv ):
        '''
        Run sparse not implemented yet
        2020/5/9
        Done
        2020/5/11
        '''

        if 'sparse' in graph.type():
            run_sparse = True
            w4_weight = self.W4.weight
            w3_weight = self.W3.weight
        else:
            run_sparse = False
        if len(graph.size()) == 2 and not run_sparse:
            graph = torch.unsqueeze(graph,  0)
    
        device = self.device
        batch_size = Xv.shape[0]
        n_vertex = Xv.shape[1]

        if not run_sparse:
            graph_edge = torch.unsqueeze(graph , 3)
        else:
            graph_edge = torch.sparse.sum(graph , 1)
            graph_edge = graph_edge.unsqueeze((1))
            #print(graph_edge)
            #saas
        emb_matrix = torch.zeros([batch_size , n_vertex , self.emb_dim ]).type(torch.DoubleTensor)
        
        if 'cuda' in Xv.type():
            if device == None:
                emb_matrix = emb_matrix.cuda()
            else:
                emb_matrix = emb_matrix.cuda(device)



        for t in range(self.T):
            #print(graph.type() , graph.shape)
            if 'sparse' in graph.type():
                neighbor_sum = torch.sparse.mm(graph , emb_matrix[0])
                neighbor_sum = neighbor_sum.view(batch_size , neighbor_sum.shape[0] , neighbor_sum.shape[1])
            else:
                neighbor_sum = torch.bmm(graph , emb_matrix )

            v1 = self.W1(Xv)
            v2 = self.W2(neighbor_sum)
            if run_sparse:
                v3 = self.relu(torch.sparse.mm(graph_edge , w4_weight.t()))
                v3 = torch.sparse.mm(v3 , w3_weight.t())
                #print(v3)
                #aaa
            else:
                v3 = self.W4(graph_edge)
                #v3 = self.relu(v3)
                v3 = self.W3(torch.sum(v3 , 2))
            v = v1 + v2 + v3
            emb_matrix = self.relu(v)

        
        emb_sum = torch.sum(emb_matrix , 1)


        
        v6 = self.W6(emb_sum)
        v6 = v6.repeat(1,n_vertex )
        v6 = v6.view(batch_size , n_vertex , self.emb_dim)
        v7 = self.W7(emb_matrix)
        ct = self.relu(torch.cat([v6 , v7] , 2))
        
        return torch.squeeze(self.W5(ct) , 2)



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
    
    def sample(self , batch_size , replace = False):
        batch = np.random.choice(np.arange(self.size) , size = batch_size , replace=replace)
        
        return self.buffer[[batch]]

    def clear_buffer(self):
        self.size = 0
        self.idx = -1

class Agent(nn.Module):
    def __init__(self , emb_dim = 64 , T = 5,device = 'cuda:0' , init_factor = 10 , w_scale = 0.01 , init_method = 'normal' , 
    replay_size = 500000 , PG = False , global_net = False , fitted_Q = True , fix_seed = True , lr = 1e-4 , adaption_test = False , weight_decay = 0.0,
    prioritize_rp = False , batch_size = 64 , pr_alpha = 0.8 , EPS_DECAY = 20000 , explore_end = 0.05):
        

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

        self.prioritize_rp = prioritize_rp
        if prioritize_rp:
            self.buffer = PrioritizedReplayBuffer(size = replay_size , alpha = pr_alpha)
            self.beta_schedule = LinearSchedule(1000000,
                                       initial_p=0.4,
                                       final_p=1.0)

        else:
            self.buffer = replay_buffer(replay_size)


        ##
        self.batch_size = batch_size
        self.adaption_test = adaption_test
        self.adaption_buffer = PrioritizedReplayBuffer(20000 , alpha = .99)
        self.adaption_schedule = LinearSchedule(20000,
                                       initial_p=0.4,
                                       final_p=1.0)
        #if adaption_test:
        #    self.train_all = False
        #    self.train_encoder = True
        #    self.train_decoder = False
        enc_parameters_list = list(self.dqn.W1.parameters()) + list(self.dqn.W2.parameters()) + list(self.dqn.W3.parameters()) + list(self.dqn.W4.parameters()) +list(self.dqn.W6.parameters())  

        dec_parameters_list = list(self.dqn.W5.parameters()) + list(self.dqn.W7.parameters())  

        self.enc_optimizer = torch.optim.Adam( enc_parameters_list , lr = lr , amsgrad=False , weight_decay = weight_decay)
        self.dec_optimizer = torch.optim.Adam( dec_parameters_list , lr = 5e-6 , amsgrad=False , weight_decay = 0.0005)
            
            #self.optimizer = torch.optim.Adam( parameters_list , lr = lr , amsgrad=False , weight_decay = weight_decay)
        #else:
        self.train_all = True
        self.optimizer = torch.optim.Adam(self.dqn.parameters() , lr = lr , weight_decay = weight_decay)

        self.loss_func = torch.nn.MSELoss()

        self.device = device

        self.steps_done = 0 
        self.adaption_steps = 0
        self.EPS_END = explore_end
        self.EPS_START = 1.0
        self.EPS_DECAY = EPS_DECAY
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

    def flip_parameter(self):
        if self.adaption_test:
            self.train_encoder = not self.train_encoder
            self.train_decoder = not self.train_decoder

    def clear_buffer(self):
        self.buffer.clear_buffer()

    def reset(self , buffer_clear = False):
        if buffer_clear:
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

            #if self.adaption_test:
            #    eps_threshold = 0.5
            
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
    def get_val_result(self, validation_graph , run_sparse = False):


        if type(validation_graph) is not list:
            validation_graph = [validation_graph]

        objective_vals = []
        for g in validation_graph:
            env = MVC_environement(g)
            Xv , graph = env.reset_env()
            if run_sparse:
                assert len(validation_graph) == 1
                #graph = torch.unsqueeze(graph,  0)
                graph = to_sparse_tensor(graph)
            else:
                graph = torch.unsqueeze(graph,  0)
            Xv = Xv.clone()
            if 'cuda' in self.device:
                Xv = Xv.cuda()
            graph = graph.to(self.device)
            done = False
            self.non_selected = list(np.arange(env.num_nodes))
            self.selected = []
            while done == False:
                #Xvprint(len(self.selected))
                #Xv = Xv.cuda()
                Xv = Xv.to(self.device)
                if run_sparse:
                    val = self.forward(graph , Xv)
                else:
                    val = self.forward(graph , Xv)[0]
                action = self.take_action(graph , Xv , is_validation = True)
                
                Xv_next , reward , done = env.take_action(action)
                Xv = Xv_next
            #print(selected)
            objective_vals.append(len(self.selected))
        return sum(objective_vals)/len(objective_vals)
    

    def batch_adaption(self , validation_graph , adap_iter = 20):
        try:
            self.adaption_steps = 0
            self.save_weights('rl_attack_model/adaption_tmp.pkl')
            self.update_target_network()

            for i in range(adap_iter):
                self.get_val_result_batch(validation_graph , during_adaption = True)



            ret_ls = self.get_val_result_batch(validation_graph , return_list = False, during_adaption = False)
            self.adaption_steps = 0
            self.adaption_buffer.clear_buffer()
            self.load_weights(file_name = 'rl_attack_model/adaption_tmp.pkl')
            return ret_ls
        finally:
            self.adaption_steps = 0
            self.adaption_buffer.clear_buffer()
            self.load_weights(file_name = 'rl_attack_model/adaption_tmp.pkl')
            
    def get_val_result_batch(self , validation_graph , return_list = False , during_adaption = False):
        N = len(validation_graph)
        N_STEP = 2
        all_graphs = []
        all_Xv = []
        all_envs = []
        

        fitted_experience_list = [[] for _ in range(N)]

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
        cur_step = 0
        while all_done == False:
            q_val = self.dqn(all_graphs , all_Xv )
            for i in range(N):

                if all_dones[i]:
                    continue

                q_val[i][ all_selected[i] ] = -float('inf')
                action = torch.argmax(q_val[i]).item()
                if during_adaption:
                    rand_val = random.uniform(0 , 1)
                    if rand_val >0:
                        probs = torch.nn.functional.softmax(q_val[i])
                        m = torch.distributions.categorical.Categorical(probs = probs)
                        action = m.sample().item()
                    #print(m,action)
                all_selected[i].append(action)
                Xv_next,rew,done = all_envs[i].take_action(action)

                if during_adaption:
                    copy_xv = deepcopy(all_Xv[i:i+1])
                    fit_ex = fitted_q_exp(all_graphs[i:i+1] , copy_xv , action , rew)
                    fitted_experience_list[i].append(fit_ex)
                    if len(fitted_experience_list[i]) >= N_STEP:
                        
                        n_reward = -N_STEP
                        n_prev_ex = fitted_experience_list[i][0]
                        n_graph = n_prev_ex.graph
                        n_Xv = n_prev_ex.Xv
                        #print( sum(n_Xv[0]) , sum(Xv_next[0]))
                        #zz
                        n_action = n_prev_ex.action
                        ex = experience(n_graph , n_Xv , torch.tensor([n_action]) , torch.tensor([n_reward]) , Xv_next , done)
                        self.adaption_buffer.push(ex)
                        #print(len(self.adaption_buffer))
                        fitted_experience_list[i].pop(0)
                    #ex = experience( all_graphs[i:i+1] , deepcopy(all_Xv[i:i+1]) , torch.tensor([action]) , torch.tensor([-1]) , Xv_next , done)
                    #self.adaption_buffer.push(ex)
                    #print(len(self.adaption_buffer))
                    #self.train(during_adaption = True , fitted_Q = True)

                all_Xv[i][action] = 1
                if done:
                    all_dones[i] = True
                    done_count += 1
                self.adaption_steps += 1
            cur_step += 1
            if during_adaption and cur_step >= N_STEP+1:
                #print(len(self.adaption_buffer))
                for ep in range(20):
                    self.train(during_adaption = True , fitted_Q = False , batch_size = 128)
                self.update_target_network()
            
            if(done_count == N):
                break
            
            #if during_adaption:
            #    for zz in range(N):
            #        self.train(during_adaption = True , fitted_Q = False)
            #        if zz % 10 == 0:
            #            self.update_target_network()
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
    
    

    def train(self   , fitted_Q = False , during_adaption = False , batch_size = None) :
        if batch_size is None:
            batch_size = self.batch_size
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

            if during_adaption:
                batch , (weights, batch_idxes) = self.adaption_buffer.sample(batch_size , beta = self.adaption_schedule.value(self.adaption_steps) )
            else:
                if self.prioritize_rp:
                    if len(self.buffer) < batch_size:
                        return
                else:  
                    if(self.buffer.size < batch_size):
                        return 
                if self.prioritize_rp:
                    batch , (weights, batch_idxes)   = self.buffer.sample(batch_size , beta = self.beta_schedule.value(self.steps_done))
                    #print(batch)
                else:
                    batch = self.buffer.sample(batch_size)
            #print(len(batch),batch)
            
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

            
            #if self.adaption_test == False:
            if not during_adaption:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                #self.optimizer.zero_grad()
                #loss.backward()
                #self.optimizer.step()
                self.dec_optimizer.zero_grad()
                loss.backward()
                self.dec_optimizer.step()
                #if during_adaption:
                #self.dec_optimizer.zero_grad()
                #loss.backward()
                #self.dec_optimizer.step()
                #if during_adaption or self.train_decoder:
                #    self.dec_optimizer.zero_grad()
                #    loss.backward()
                #    self.dec_optimizer.step()
                #else:
                #    self.enc_optimizer.zero_grad()
                #    loss.backward()
                #    self.enc_optimizer.step()

            if self.prioritize_rp and not during_adaption:
                td_errors =  np.abs(expected_q.cpu().detach() - pred_q.cpu().detach()).numpy()
                new_priorities = td_errors + 1e-5
                #print(new_priorities)
                self.buffer.update_priorities(batch_idxes , new_priorities)
            
            if during_adaption:
                td_errors =  np.abs(expected_q.cpu().detach() - pred_q.cpu().detach()).numpy()
                new_priorities = td_errors + 1e-5
                self.adaption_buffer.update_priorities(batch_idxes , new_priorities)

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

                #print(sum(n_Xv[0]) , sum(Xv_next[0]))
                #aa
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