from matplotlib.offsetbox import AnchoredText
from node2vec.node2vec import node2vec
from sbm.sbm import stochastic_block_model
import datetime
import networkx as nx
import itertools
import glob
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.sparse as sp
import time
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

def get_name_to_date(G):
    """
    get dictionary mapping from doc_name to year published

    Parameters
    ----------
    G, the network object of (probably) SCOTUS dataset

    Output
    ------
    dict with keys being name, values being date in year
    """
    dates = {}
    for v in G.nodes():
        date = float(str(G.node[v]["year"]).strip())//1
        dates[G.node[v]["name"]] = int(date)
    return dates

def get_list_of_docs(dir_path='../data/scotus/textfiles/*.txt'):
    """
    get list of documents as strings from directory

    Parameters
    ----------
    path to directory of .txt files

    Output
    ------
    list of docs, list of names
    """
    # try to accept different variations/understings of dir_path
    if dir_path[-1]!='/' and dir_path[-4:]!='.txt':
        dir_path = dir_path + '/*.txt'
    elif dir_path[-4:]!='.txt':
        dir_path = dir_path + '*.txt'

    files = glob.glob(dir_path)
    documents = []
    names = []

    for filename in files:
        with open(filename, 'r', encoding="utf8") as readfile:
            names.append(os.path.basename(filename)[:-4])
            documents.append(readfile.read())
    return documents, names

def stratified_sample_split(class_to_id={},proportion_in_test=0.2):
    """
    Stratified Random Sample into train/test split on SCOTUS dataset

    Parameters
    ----------
    dict-like object with key being class_id,
        and the value being a list of ids corresponding to that class 

    Output
    ------
    np.array of train_indices, np.array of test_indices
    """
    print('FUNCTION NOT FULLY TESTED: stratified_sample_split')
    sampled_indices = np.array([])
    non_sampled_indices = np.array([])

    sampled_indices_dict = {i:None for i in range(len(class_to_id.keys()))}

    non_sampled_indices_dict = {i:None for i in range(len(class_to_id.keys()))}

    for i in iter(class_to_id):
        msk = np.random.rand(len(class_to_id[i])) >= proportion_in_test
        
        tmp_sampled = np.array(
            [class_to_id[i][j] for j in range(len(msk)) if msk[j]==True])
        tmp_non_sampled = np.array(
            [class_to_id[i][j] for j in range(len(msk)) if msk[j]==False])
        
        sampled_indices = np.append(sampled_indices, tmp_sampled)
        non_sampled_indices = np.append(non_sampled_indices, tmp_non_sampled)

        sampled_indices_dict[i] = sampled_indices
        non_sampled_indices_dict[i] = non_sampled_indices
            
    sampled_indices = np.array(np.sort(sampled_indices,axis=None),dtype=int)
    non_sampled_indices = np.array(np.sort(non_sampled_indices,axis=None),dtype=int)
    print("Total length of training indices: "+str(len(sampled_indices))+"\n"+str(sampled_indices)+"\n")
    print("Total length of test indices: "+str(len(non_sampled_indices))+"\n"+str(non_sampled_indices))
    return sampled_indices, non_sampled_indices

def load_scotus_network(file_path="../data/scotus/scotus_network.graphml",undirected=True,relabel=True,force_issueArea=True):
    """
    load the networkx network for the scotus dataset

    Parameters
    ----------
    file_path: location of network

    Output
    ------
    graph object; dict like : {id : issueArea}
    """
    print('FUNCTION NOT FULLY TESTED: load_scotus_network')
    if '.graphml' in file_path:
        G = nx.read_graphml(file_path)
    elif '.adjlist' in file_path:
        G = nx.read_adjlist(file_path)
    else:
        raise ValueError("Unsupported reading filetype")
    if undirected:
        G = G.to_undirected()
    if relabel:
        relabel_mapping = {}
        for node_id in G.nodes():
            relabel_mapping[node_id] = G.node[node_id]['name']
        G = nx.relabel_nodes(G, relabel_mapping, copy=True)
    if force_issueArea:
        issueAreas = {node_id : int(G.node[node_id]['issueArea']) for node_id in G.nodes()}
    else:
        issueAreas = {node_id : int(G.node[node_id].get('issueArea',0)) for node_id in G.nodes()}
    return G, issueAreas

def load_scotus_tf_idf(file_path="../data/local/scotus/tfidf_matrix.npz"):
    """
    load the sparse tf-idf matrix for the scotus dataset

    Parameters
    ----------
    file_path: location of .npz

    Output
    ------
    scipy sparse csr_matrix
    """
    tf_idf = np.load(file_path)
    data = tf_idf.f.data
    indices = tf_idf.f.indices
    indptr = tf_idf.f.indptr
    data_shape = tuple(tf_idf.f.shape)
    return sp.csr_matrix((data, indices, indptr),shape=data_shape)

def score_bhamidi(memberships, predicted_memberships):
    """
    Scores the predicted clustering labels vs true labels using Bhamidi's scoring method.

    Parameters
    ----------
    memberships: actual true labels
    predicted_memberships: clustering labels

    Output
    ------
    a percentage of correct vertice pairings
    """
    score = 0
    vertex_count = len(memberships)
    for i in range(vertex_count):
        for j in range(vertex_count):
            actual_match = memberships[i]==memberships[j]
            predicted_match = predicted_memberships[i]==predicted_memberships[j] 
            if actual_match == predicted_match:
                score += 1
    #convert to percent of total vertex pairs
    score = score/(vertex_count*vertex_count)
    return score


def score_purity(memberships, predicted_memberships):
    """
    Scores the predicted clustering labels vs true labels using purity.

    Parameters
    ----------
    memberships: actual true labels
    predicted_memberships: clustering labels

    Output
    ------
    a percentage of correct labels
    """
    num_nodes = len(memberships)
    
    #identify unique labels
    true_labels = set(memberships)
    predicted_labels = set(predicted_memberships)
    
    #make a set for each possible label
    true_label_sets = {}
    predicted_label_sets = {}
    for label in true_labels:
        true_label_sets[label] = set()
        predicted_label_sets[label] = set()
    
    #go through each vertex and assign it to a set based on label
    for i in range(num_nodes):
        true_label = memberships[i]
        predicted_label = predicted_memberships[i]
        true_label_sets[true_label].add(i)
        predicted_label_sets[predicted_label].add(i)
        
    #now can perfrom purity algorithm
    score = 0
    for true_label, true_label_set in true_label_sets.items():
        max_intersection = 0
        for predicted_label, predicted_label_set in predicted_label_sets.items():
            intersection = len(set.intersection(predicted_label_set,true_label_set))
            if max_intersection < intersection:
                max_intersection = intersection
        score += max_intersection

    #divide score by total vertex count
    score = score/num_nodes
    return score

def score_agreement(y, y_hat):
    '''
    calculates agreement score for the labeling of a community

    input variables
    x - true labels of verticies
    y - predicted labels of verticies
    output variable
    score - agreement score
    '''
    max_score = 0
    relabelings = create_relabelings(y_hat)
    y = np.array(y)
    return max([np.sum(y == relabeling)/len(y) for relabeling in relabelings])

def score_auc(x,y):
    '''
    calculates the area under the curve of the x,y pairs using the Trapezoidal rule
    x - vector of x values
    y - vector of y values
    '''
    return np.trapz(y,x=x)

def create_relabelings(y):
    '''
    calculates possible of permutations of labels of predicted labels of verticies
    i.e. maintain community association within vector but changes the label values

    input variables
    y - relabelings of vertices
    output variable
    relabeling - possible relabelings of y
    '''
    lookups = list(set(itertools.permutations(set(y))))
    relabelings = []
    for lookup in lookups:
        lookup = np.array(lookup)
        relabelings.append(np.array(lookup[y]))
    return relabelings

def make_block_probs(in_class_prob=0.5, out_class_prob=0.5):
    return np.array([[in_class_prob, out_class_prob],
                     [out_class_prob, in_class_prob]])

def multiple_sbm_iterate(start = 30,
                        stop = 91,
                        step = 2,
                        walk_length = 50,
                        num_walks = 25,
                        num_nodes = 400,
                        n_classes = 2,
                        in_class_prob = 0.8,
                        iterations = 10,
                        p = 1.0,
                        q = 1.0,
                        samples = 10):
    print('At multiple_sbm_iterate(...)')
    start_time = time.clock()
    # will be y-axis on plot
    bhamidi_scores_plot = []
    bhamidi_medians = []
    purity_scores_plot = []
    purity_medians = []
    agreement_scores_plot = []
    agreement_medians = []
    # will be x-axis on plot
    out_class_probs = []
    
    first_iter = True
    for i in range(samples):
        tmp_bhamidi_scores_plot,\
        _0,\
        tmp_purity_scores_plot,\
        _1,\
        tmp_agreement_scores_plot,\
        _2,\
        tmp_out_class_probs = iterate_out_of_class_probs(start = start,
                                        stop = stop,
                                        step = step,
                                        walk_length = walk_length,
                                        num_walks = num_walks,
                                        num_nodes = num_nodes,
                                        n_classes = n_classes,
                                        in_class_prob = in_class_prob,
                                        iterations = iterations,
                                        p = p,
                                        q = q)

        if first_iter:
            bhamidi_scores_plot = tmp_bhamidi_scores_plot
            purity_scores_plot = tmp_purity_scores_plot
            agreement_scores_plot = tmp_agreement_scores_plot
            out_class_probs = tmp_out_class_probs
            first_iter = False
        else:
            bhamidi_scores_plot = [bhamidi_scores_plot[j]+tmp_bhamidi_scores_plot[j] for j in range(len(bhamidi_scores_plot))]
            purity_scores_plot = [purity_scores_plot[j]+tmp_purity_scores_plot[j] for j in range(len(purity_scores_plot))]
            agreement_scores_plot = [agreement_scores_plot[j]+tmp_agreement_scores_plot[j] for j in range(len(agreement_scores_plot))]
            
    bhamidi_medians = [np.median(bhamidi_scores_plot[j]) for j in range(len(bhamidi_scores_plot))]
    purity_medians = [np.median(purity_scores_plot[j]) for j in range(len(purity_scores_plot))]
    agreement_medians = [np.median(agreement_scores_plot[j]) for j in range(len(agreement_scores_plot))]
    print("Time elapsed while running 'multiple_sbm_iterate' function: {0}".format(round(time.clock()-start_time,8)))
    return bhamidi_scores_plot, bhamidi_medians, purity_scores_plot, purity_medians, agreement_scores_plot, agreement_medians, out_class_probs
    

def eval_multiple_walks(sbm, w_length=50, n_classes=2, num_walks=25, p=1, q=1, iterations=5):
    '''
    Return the bhamidi, purity, and agreement scores after sampling node2vec walks for the specified number of iterations

    Parameters
    ------------
    sbm : stochastic block matrix from which the graph object should be defined
    w_length : length of node2vec walk
    n_classes : number of classes; also number of clusters because we will use kmeans to evaluate
    num_walks : number of node2vec walks to generate per node in graph object
    p : Return parameter; lower values result in more "local" walks
    q : In-out parameter; lower values result in more Depth-First Search behaving walks
    iterations : number of times the node2vec walks should be regenerated, understanding that the node embeddings must
                    be recalculated every time the walks are regenerated
    '''
    print('At eval_multiple_walks(...)')
    start_time = time.clock()
    bhamidi_scores = []
    purity_scores = []
    agreement_scores = []
    for i in range(iterations):
        node_embeds = node2vec(G=None,
                                Adj_M=sbm.A,
                                labels=sbm.memberships,
                                n_classes=n_classes,
                                evaluate=True,
                                p=p,
                                q=q,
                                walk_length=w_length,
                                num_walks=num_walks,
                                window_size=10,
                                embedding_size=128,
                                num_iter=4,
                                min_count=0,
                                sg=1,
                                workers=8,
                                )
        bhamidi_scores.append(node_embeds.bhamidi_score)
        purity_scores.append(node_embeds.purity_score)
        agreement_scores.append(node_embeds.agreement_score)
    print("Time elapsed while running 'eval_multiple_walks' function: {0}".format(round(time.clock()-start_time,8)))
    # both are of type : list
    return bhamidi_scores, purity_scores, agreement_scores

def iterate_out_of_class_probs(start = 30,
                               stop = 91,
                               step = 2,
                               walk_length = 50,
                               num_walks = 25,
                               num_nodes = 400,
                               n_classes = 2,
                               in_class_prob = 0.8,
                               iterations = 10,
                               p = 1.0,
                               q = 1.0):
    print('At iterate_out_of_class_probs(...)')
    start_time = time.clock()
    # will be y-axis on plot
    bhamidi_scores_plot = []
    bhamidi_medians = []
    purity_scores_plot = []
    purity_medians = []
    agreement_scores_plot = []
    agreement_medians = []
    # will be x-axis on plot
    out_class_probs = []

    iteration_counter = 1
    for i in range(start, stop, step):
        # keep track of where the program is:
        if iteration_counter%5==0:
            print('Currently at iteration : {0}'.format(iteration_counter))

        iteration_counter += 1
        # change i into a probability
        i *= 0.01
        # i will become out_class_prob
        # in_class_prob is static
        block_probs = make_block_probs(in_class_prob=in_class_prob, out_class_prob=i)
        sbm = stochastic_block_model(size=num_nodes,
                                     block_probabilities=block_probs,
                                     num_classes=n_classes)
        bhamidi_scores, purity_scores, agreement_scores = eval_multiple_walks(sbm,
                                                            w_length=walk_length,
                                                            n_classes=n_classes,
                                                            num_walks=num_walks,
                                                            p=p,
                                                            q=q,
                                                            iterations=iterations)
        # record for plotting purposes
        bhamidi_scores_plot.append(bhamidi_scores)
        bhamidi_medians.append(np.median(bhamidi_scores))
        purity_scores_plot.append(purity_scores)
        purity_medians.append(np.median(purity_scores))
        agreement_scores_plot.append(agreement_scores)
        agreement_medians.append(np.median(agreement_scores))
        out_class_probs.append(i)
    print("Time elapsed while running 'iterate_out_of_class_probs' function: {0}".format(round(time.clock()-start_time,8)))
    return bhamidi_scores_plot, bhamidi_medians, purity_scores_plot, purity_medians, agreement_scores_plot, agreement_medians, out_class_probs

def save_current_status(file_name = 'current_status',**kwargs):
    # saving current status
    kwargs.pop('file_name')
    # save to file (as json, obviously)
    with open(file_name+'.json', 'w') as fp:
        json.dump(current_status, fp)
    return current_status

def plot_save_scores(out_class_probs=[],
                     bhamidi_scores_plot=None,
                     purity_scores_plot=None,
                     agreement_scores_plot=None,
                     bhamidi_medians=None,
                     purity_medians=None,
                     agreement_medians=None,
                     file_name='current_status',
                     walk_length = 'N/a',
                     num_walks = 'N/a',
                     num_nodes = 'N/a',
                     n_classes = 'N/a',
                     in_class_prob = 'N/a',
                     iterations = 'N/a',
                     p = 'N/a',
                     q = 'N/a',**kwargs):
    plt.style.use('ggplot')
    # first plot : plot scores
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title('Explore Phase Change: Resampling Walks & SBM',color='black',fontsize=18)
    ax.set_ylabel('Scores',color='black',fontsize=14)
    ax.set_xlabel('Out-Class Probability',color='black',fontsize=14)
    ax.tick_params(axis='both',color='black')
    # set ticks to be the color black
    plt.setp(ax.get_xticklabels(), color='black')
    plt.setp(ax.get_yticklabels(), color='black')

    # plot scores as scatter plot first
    for i in range(len(out_class_probs)):
        x = out_class_probs[i]

        if isinstance(bhamidi_scores_plot,list):
            # bhamidi scores plot
            y = bhamidi_scores_plot[i]
            if i == 0:
                ax.scatter([x]*len(y), y, alpha=0.4, marker='.', c='g', label='raw bhamidi scores')
            else:
                ax.scatter([x]*len(y), y, alpha=0.4, marker='.', c='g')
        
        if isinstance(purity_scores_plot,list):
            # purity scores plot
            y = purity_scores_plot[i]
            if i == 0:
                ax.scatter([x]*len(y), y, alpha=0.4, marker='.', c='b', label='raw purity scores')
            else:
                ax.scatter([x]*len(y), y, alpha=0.4, marker='.', c='b')

        if isinstance(agreement_scores_plot,list):
            # agreement scores plot
            y = agreement_scores_plot[i]
            if i == 0:
                ax.scatter([x]*len(y), y, alpha=0.4, marker='.', c='r', label='raw agreement scores')
            else:
                ax.scatter([x]*len(y), y, alpha=0.4, marker='.', c='r')

    ax.xaxis.set_ticks(np.arange(0.0, 1.1, 0.1)) # up to, but not including 1.1
    ax.yaxis.set_ticks(np.arange(0.5, 1.1, 0.1)) # up to, but not including 1.1
    
    if isinstance(bhamidi_scores_plot,list):
        median_bhamidi, = ax.plot(out_class_probs, bhamidi_medians, '-', color='g', label='median bhamidi scores')
    if isinstance(purity_scores_plot,list):
        median_purity, = ax.plot(out_class_probs, purity_medians, '-', color='b', label='median purity scores')
    if isinstance(agreement_scores_plot,list):
        median_agreement, = ax.plot(out_class_probs, agreement_medians, '-', color='r', label='median agreement scores')
    # create the legend
    legd = ax.legend(loc=3,fancybox=True,fontsize=12,scatterpoints=3)
    for text in legd.get_texts():
        text.set_color('black')
    anchored_text = AnchoredText('''---------PARAMS---------
walk length : {0}
num of walks : {1}
num of nodes : {2}
num of classes : {3}
in-class prob. : {4}
iterations : {5}
p : {6}
q : {7}'''.format(walk_length, num_walks, num_nodes, n_classes, round(in_class_prob,2), iterations, round(p,2), round(q,2)),loc=7, bbox_to_anchor=(1, 0.5))
    ax.add_artist(anchored_text)
    plt.savefig(file_name+'.png')
    plt.show()
