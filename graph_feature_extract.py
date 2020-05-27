import networkx as nx
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_distances as cos_distance
from sklearn.cluster import KMeans
from random import shuffle
from collections import defaultdict

def get_mean_var(arr):
    return np.mean(arr) , np.var(arr) , np.min(arr) , np.percentile(a = arr , q = 25) , np.percentile(a = arr , q = 50) , np.percentile(a = arr , q = 75) , np.max(arr)

def linear_maximum_diversity_select(matrix , k , method = 'linear'):
    if method == 'linear':
        pair_dist = eu_distance(matrix)
        ret_arr = []
        cnt = 0
        cur = 0
        while cnt < k:
            ret_arr.append(cur)
            cur_dist = pair_dist[cur]
            cur_dist[ret_arr] = -1
            cur = np.argmax(cur_dist)
            cnt += 1
        return ret_arr
    elif method == 'kmeans':
        kmeans = KMeans(n_clusters = k , max_iter = 8000 ,tol = 1e-6)
        kmeans.fit(matrix)
        center = kmeans.cluster_centers_
        label_idx = defaultdict(list)
        for i , l in enumerate(kmeans.labels_):
            label_idx[l].append(i)
        ret_arr = []
        for k , v in label_idx.items():
            shuffle(v)
            ret_arr.append(v[0])
        return ret_arr

complete_g = None
def iterative_add_edge(g1 , B , C = 200 , method = 'sorted'):
    global complete_g 

    if complete_g == None:
        complete_g = nx.complete_graph(n = len(g1.nodes()))

    added_edges = complete_g.edges() - g1.edges()
    g2 = deepcopy(g1)

    if method == 'sorted':
        uv_scr = []
        #print(len(g2.edges()))
        added_edges = list(added_edges)
        existed_edges = list(g1.edges())
        random.shuffle(added_edges)
        
        for e in existed_edges[:C]:
            u , v = e
            g2.remove_edge(u , v)
            cos , eu = get_two_graph_distance(g1,g2)
            uv_scr.append([(u,v),cos]) 
            g2.add_edge(u,v)
            
        for e in added_edges[:C]:
            u , v = e
            
            g2.add_edge(u,v)
            cos , eu = get_two_graph_distance(g1,g2)
            uv_scr.append([(u,v),cos]) 

            g2.remove_edge(u,v)
        uv_scr.sort(key = lambda s : -s[1] )
        #print(len(uv_scr),uv_scr)
        alread_added = set()
        
        cnt = 0
        
        for e in uv_scr[:]:
            (u , v) , sc = e
            if u in alread_added or v in alread_added:
                continue
            if (u,v) not in g2.edges():
                g2.add_edge(u , v)
            else:
                g2.remove_edge(u , v)
            alread_added.add(u)
            alread_added.add(v)
            cnt += 1
            if cnt >= B:
                break
        
        
        return g2
    else:
        for _ in range(B):
            mx = 0.0
            mxuv = (-1,-1)
            for e in added_edges:
                u , v = e
                g2.add_edge(u,v)

                cos , eu = get_two_graph_distance(g1,g2)
                if cos > mx:
                    mx = cos
                    mxuv = u , v

                g2.remove_edge(u,v)
            g2.add_edge(mxuv[0] , mxuv[1])
            added_edges.remove(mxuv)
        return g2
def get_two_graph_distance(g1 , g2):
    f1 = np.array(get_graph_features(g1)).reshape(1,-1)
    f2 = np.array(get_graph_features(g2)).reshape(1,-1)
    return cos_distance(f1,f2).item() , eu_distance(f1,f2).item()
    

def get_graph_features(g):

    feature_list = []

    coloring_num = max(nx.algorithms.coloring.greedy_color(g).values())

    for _ in range(7):
        feature_list.append(coloring_num)

    cluestering_arr = np.array(list(dict(nx.algorithms.cluster.clustering(g)).values()))
    features = get_mean_var(cluestering_arr)
    for f in features:
        feature_list.append(f)

    deg_arr = np.array(list(dict(nx.degree(g)).values()))
    features = get_mean_var(deg_arr)

    for f in features:
        feature_list.append(f)


    core_arr = np.array(list(dict(nx.algorithms.core.core_number(g)).values()))
    features = get_mean_var(core_arr)
    for f in features:
        feature_list.append(f)

    #feature_list = [coloring_num , cluestering_mean , cluestering_var , deg_mean , deg_var , core_mean , core_var]

    return feature_list




def get_graphs_normalized_features(graphs):

    all_features = []
    for g in graphs:
        all_features.append( get_graph_features(g))
    all_features = np.array(all_features)
    #print(all_features)
    #print(normalize(all_features,axis=0 , norm='l1'))

    return normalize(all_features,axis=0 , norm='l1')