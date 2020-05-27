from sklearn.metrics.pairwise import euclidean_distances as eu_distance
from graph_feature_extract import get_graph_features , get_graphs_normalized_features
import networkx as nx
import numpy as np

def get_pairwiseDistance(graphs ):
    N = len(graphs)
    matrix = get_graphs_normalized_features(graphs)
    pair_dist = eu_distance(matrix)
    return pair_dist

def get_pairwiseDistance_graph(graphs , tau = None):
    N = len(graphs)
    matrix = get_graphs_normalized_features(graphs)
    pair_dist = eu_distance(matrix)
    
    if tau is None:
        vec = pair_dist.reshape(-1)
        tau = np.mean(vec)
    
    ret_g = nx.Graph()
    
    ret_g.add_nodes_from([ i  for i in range(N)])
    
    for i in range(N):
        for j in range(i,N):
            if pair_dist[i][j] >= tau:
                ret_g.add_edge(i,j , w = pair_dist[i][j])
                ret_g.add_edge(j,i,  w = pair_dist[i][j])
    
    return ret_g



def color_sort(graph , R):
    '''
    colors = nx.greedy_color(graph)
    colors_list = []
    for k , v in colors.items():
        colors_list.append([k,v])
    
    colors_list.sort(key = lambda c : c[1])
    
    vertices_order , vertices_color = [] , {}
    for v , c in colors_list:
        vertices_order.append(v)
        vertices_color[v] = c
    
    return vertices_order , vertices_color
    '''
    Ck = [[]]
    max_no = 0
    vert_color = {}
    for i in range(len(R)):
        p = R[i]
        k = 0
        while k < len(Ck) and  len( list_intersection(Ck[k] , list(graph.neighbors(p))) ) > 0 :
            k = k + 1
        
        if k > max_no:
            max_no = k
            Ck.append([])
        vert_color[p] = k
        Ck[k].append(p)
        
    sorted_vertices = []
    for i in range(max_no + 1):
        for v in Ck[i]:
            sorted_vertices.append(v)
    return sorted_vertices , vert_color



def list_intersection(a , b):
    return list(set(a)&set(b))

def get_diversity_sum(graph , sol):
    ret = 0
    subg = graph.subgraph(sol)
    for v in sol:
        for vv , k in dict(subg[v]).items():
            #print(vv,k)
            ret += k['w']
    return ret

best_sol = []
obj_upper_bound = 0
cur_largest_clique = []


def best_tau_selection(graphs , k , init_sort = 'weight'):

    up = 1.0
    down = 0.0
    ret_best_sol = []

    while (up - down > 0.0001):
        mid = (up + down) / 2
        print('cur tau' , mid)
        sol = kclique_selection(graphs , k  , init_sort, mid)
        ret_best_sol = sol[:]
        
        if len(sol) == k:
            break
            #return sol

        elif len(sol) < k:
            up = mid
        else:
            down = mid
    print('select {} graphs'.format(len(ret_best_sol)))
    return ret_best_sol



def kclique_selection(graphs , k , init_sort = 'weight', tau = None):
    global best_sol
    global obj_upper_bound
    global cur_largest_clique
    best_sol = []
    obj_upper_bound = 0
    cur_largest_clique = []

    kclique_g = get_pairwiseDistance_graph(graphs , tau = tau)

    if max(dict(nx.core_number(kclique_g)).values()) < k-1:
        return []

    #vert_deg = []
    #for v , d in dict(nx.degree(kclique_g)).items():
    #    vert_deg.append([v,d])
        
    #vert_deg.sort(key = lambda v : v [1], reverse=True)

    vert_edgeWeight = []
    for v in kclique_g.nodes():
        s = 0
        d = 0
        for u , w in dict(kclique_g[v]).items():
            s += w['w']
            d += 1
        vert_edgeWeight.append([v,s,d])
    if init_sort == 'weight':
        vert_edgeWeight.sort(key = lambda v : v[1])
    else:
        vert_edgeWeight.sort(key = lambda v : v[2])
    #vert_edgeWeight

    R = []
    for v , w,d in vert_edgeWeight:
        R.append(v)


    candidates , vertices_color = color_sort(kclique_g , R)
    kclique_maximal_edgeWeight(kclique_g , [] , k   ,candidates,vertices_color)
    return best_sol

def kclique_maximal_edgeWeight(graph , cur_sol , k   , candidates  , vertices_color ):
    global best_sol
    global obj_upper_bound
    global cur_largest_clique
    '''
    if len(cur_sol) == k:
        cur_sum = get_diversity_sum(graph , cur_sol)
        if cur_sum > obj_upper_bound:
            obj_upper_bound = max(obj_upper_bound , cur_sum)
            best_sol = cur_sol[:] 
        return 
    '''
    #print(len(cur_sol))
    while(len(candidates) > 0):

        v = candidates[-1]
        
        #print('cur sol',cur_sol)
        #print('v',v)
        if  (len(cur_sol) + vertices_color[v] > len(cur_largest_clique) ) and len(cur_sol) < k:

            
            new_cand = list_intersection( list( graph.neighbors(v) ) , candidates  )
        
            cur_sol.append(v)
            if len(new_cand) > 0:
                new_cand_order , new_vert_color = color_sort(graph , new_cand)

                kclique_maximal_edgeWeight(graph , cur_sol , k  , new_cand_order , new_vert_color)
            
                
            
            
            
            elif ( len(cur_sol) >= len(cur_largest_clique) ):
                cur_sum = get_diversity_sum(graph , cur_sol)
                
                if cur_sum > obj_upper_bound:
                    obj_upper_bound = max(obj_upper_bound , cur_sum)
                    best_sol = cur_sol[:] 
                
                    cur_largest_clique = cur_sol[:]
                    print('best ',len(cur_largest_clique))

            cur_sol.pop(-1)

        else:
            
            return 

        candidates.pop(-1)
        if len(cur_largest_clique) >= k:
            break
        