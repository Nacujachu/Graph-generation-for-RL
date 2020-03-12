import numpy as np
import networkx as nx
import torch
import scipy as sc
class MVC_environement():
    def __init__(self , nx_graph):
        
        self.nx_graph = nx_graph
        self.num_nodes = len(nx_graph.nodes())
        self.num_edegs = len(nx_graph.edges())
        #self.Xv = np.zeros([1,self.num_nodes])
        self.Xv = np.zeros([1,self.num_nodes,1])
        self.edges = list(nx_graph.edges())
        self.covered_edges = 0
        self.covered_set = set()
        self.adj_list = self.get_adj_list()
    
    def reset_env( self , sparse = False ):
        self.Xv = np.zeros([1,self.num_nodes,1])
        if sparse:
            g = self.get_torch_sparse()
            return torch.from_numpy(self.Xv).type(torch.DoubleTensor) , g
        else:
            g = self.get_torch_dense()
            return torch.from_numpy(self.Xv).type(torch.DoubleTensor) , g

    def get_lap_torch(self):
        return torch.from_numpy(nx.normalized_laplacian_matrix(self.nx_graph).toarray()).unsqueeze(0)
    
    def take_action(self,v):
        self.Xv[0,v] = 1
        self.covered_set.add(v)
        neighbor = self.adj_list[v]
        
        for u in neighbor:
            if u not in self.covered_set:
                self.covered_edges += 1
        
        DONE = False
        
        if self.covered_edges == self.num_edegs:
            DONE= True
        
        return torch.from_numpy(self.Xv).type(torch.DoubleTensor) , -1 , DONE 
        
        
    def get_adj_list(self):
        adj_list = {}
        for k , v in dict(self.nx_graph.adjacency()).items():
            adj_list[k] = []
            for vv in v.keys():
                adj_list[k].append(vv)
        return adj_list
    def get_matrix(self , sparse = True):
        if sparse == True:
            ret_g = nx.convert_matrix.to_scipy_sparse_matrix(self.nx_graph)
            ret_g = ret_g.tocoo()
            return ret_g
        else:
            ret_g = nx.convert_matrix.to_numpy_array(self.nx_graph)
            return ret_g
    
    def get_torch_dense(self):
        de = self.get_matrix(sparse = False)
        de = torch.from_numpy(de).type(torch.DoubleTensor)
        return de

    def get_torch_sparse(self):
        sp = self.get_matrix(sparse = True)
        index = np.vstack([sp.row , sp.col] ).astype(int)
        value = sp.data.astype(float)
        sz = sp.shape
        i = torch.LongTensor(index)
        v = torch.FloatTensor(value)
        return torch.sparse.FloatTensor(i, v, torch.Size(sz))