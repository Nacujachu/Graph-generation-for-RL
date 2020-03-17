from heapdict import heapdict
import numpy as np
import networkx as nx
import pickle
import random
import math
import torch
def pickle_save(data,file_name):
    with open(file_name,'wb') as f:
        pickle.dump(data , f)
def pickle_load(file_name):
    with open(file_name,'rb') as f:
        return pickle.load(f)




def density_to_edge_ba(n , p):
    return math.ceil( p *n  /2)
def validation_graph_gen(n , p , num = 100, graph_type = 'er'):
    np.random.seed(19960214)
    random.seed(19960214)
    validation_graph = []
    for i in range(num):
        if graph_type == 'er':
            g = nx.erdos_renyi_graph(n , p)
            validation_graph.append(g)
        elif graph_type == 'ba':
            m = density_to_edge_ba(n , p)
            g = nx.barabasi_albert_graph(n , m)
            validation_graph.append(g)
            
    return validation_graph

def get_lap_torch(g):
    return torch.from_numpy(nx.normalized_laplacian_matrix(g).toarray()).unsqueeze(0)

def validation(dqn , validation_graph , device = 'cuda:0'):
    objective_vals = []
    for g in validation_graph:
        env = MVC_environement(g)
        Xv , graph = env.reset_env()
        graph = torch.unsqueeze(graph,  0)
        Xv = Xv.clone()
        Xv = Xv.cuda()
        graph = graph.to(device)
        done = False
        non_selected = list(np.arange(env.num_nodes))
        selected = []
        while done == False:
            #Xv = Xv.cuda()
            Xv = Xv.to(device)
            val = dqn(graph , Xv)[0]
            val[selected] = -float('inf')
            action = int(torch.argmax(val).item())
            Xv_next , reward , done = env.take_action(action)
            non_selected.remove(action)
            selected.append(action)
            Xv = Xv_next
        #print(selected)
        objective_vals.append(len(selected))
    return sum(objective_vals)/len(objective_vals)

def is_vertex_cover(graph , cover):
    cover_edge = 0
    total_edge = len(graph.edges())
    checked_set = set()
    for c in cover:
        checked_set.add(c)
        for u in list(graph.neighbors(c)):
            if u not in checked_set:
                cover_edge += 1
            
    return cover_edge == total_edge

def mvc_bb(graph , UB = 9999999 , C = []):
    
    def DegLB():
    
        degree_list = nx.degree(graph)
        hd = heapdict()
        for v , d in degree_list:
            hd[v] = -d
        
        num_edge = len(graph.edges())
        count_edge = 0
        lb = 0
        select_nodes = []
        while count_edge < num_edge:
            cur_v , cur_degree = hd.popitem()
            count_edge += -(cur_degree)
            select_nodes.append(cur_v)
            for u in graph.neighbors(cur_v):
                if u in select_nodes:
                    continue
                hd[u] += 1
        cur_v , cur_degree = hd.popitem()
        graph_plum = graph.copy()
        graph_plum.remove_nodes_from(select_nodes)
        E_plum = len(graph_plum.edges())
        if cur_degree > 0:
            E_plum /= cur_degree
            
        return math.floor( len(select_nodes) + E_plum )
    
    if len(graph.edges()) == 0:
        #return len(C)
        return C
    #LB = DegLB()
    LB = 0
    if len(C) + LB >= UB:
        #return UB
        return [i for i in range(UB+1)]
    
    degree_list = nx.degree(graph)
    v , d = max(degree_list , key = lambda a : a[1])
    
    C1 = C[:]
    C2 = C[:]
    graph_1 = graph.copy()
    C1.extend(list(graph.neighbors(v)))
    #C1.append(v)
    graph_1.remove_nodes_from(C1)
    C1= mvc_bb(graph_1 , UB , C1)

    C2.append(v)
    graph_2 = graph.copy()
    graph_2.remove_node(v)
    C2 = mvc_bb(graph_2 , min(UB , len(C1)) , C2 )

    if len(C1)>len(C2):
        return C2
    else:
        return C1
