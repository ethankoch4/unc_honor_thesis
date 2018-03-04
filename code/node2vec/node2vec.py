import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import sys
sys.path.append('../')
import numpy as np
import networkx as nx
from node2vec.graph import Node2VecGraph
from gensim.models.word2vec import Word2Vec

# personal preference
np.set_printoptions(precision=3, suppress=True)


class node2vec(object):
    def __init__(
            self,
            G=None,
            Adj_M=None,
            n_classes=None,
            evaluate=False,
            p=1,
            q=1,
            walk_length=50,
            num_walks=25,
            window_size=10,
            embedding_size=300,
            num_iter=100,
            min_count=0,
            sg=1,
            workers=24,
            ):
        if G:
            nx_G = G
        else:
            nx_G = self.read_graph(Adj_M)
        self.G = Node2VecGraph(nx_G, False, p, q)
        
        if Adj_M is None:
            self.Adj_M = nx.to_numpy_matrix(nx_G)
        else:
            self.Adj_M = Adj_M
            
        self.evaluate = evaluate
        self.n_classes = n_classes
        self.num_walks = num_walks
        self.walk_length = walk_length

    def run_node2vec(self, labels_dict={}):
        self.G.preprocess_transition_probs()

        self.walks = self.G.simulate_walks(self.num_walks, self.walk_length)
        
        self.model = self.learn_embeddings(self.walks)

        if self.evaluate:
            self.run_clustering(self.model,
                                labels_dict=labels_dict,
                                n_clusters=self.n_classes)
        return self.model

    def read_graph(self, Adj_M):
        # only support undirected graphs as of now
        G = nx.from_numpy_matrix(Adj_M)
        G = G.to_undirected()
        return G

    def learn_embeddings(self, walks, window_size=10, alpha=0.025,
                         embedding_size=300, num_iter=100, min_count=0, sg=1,
                         workers=24, min_alpha=0.001, negative=10, sample=1e-5):
        '''
        Learn embeddings by optimizing the Skipgram objective using SGD.
        '''
        walks = [list(map(str, walk)) for walk in walks]
        return Word2Vec(walks, alpha=alpha, size=embedding_size, window=window_size,
                        min_count=min_count, sg=sg, workers=workers, iter=num_iter,
                        min_alpha=min_alpha, negative=negative, sample=sample)

    def kmeans_evaluate(self, embeddings, node_ids=[], n_clusters=2):
        from sklearn.cluster import KMeans

        if not node_ids == []:
            walks_data = embeddings.wv
            walks = []
            for node_id in node_ids:
                walks.append(walks_data[node_id])
            del walks_data

            kmeans = KMeans(n_clusters=n_clusters).fit(walks)
            return kmeans
            
    def hierarchical_evaluate(self, embeddings, node_ids=[], n_clusters=2):
        from sklearn.cluster import AgglomerativeClustering

        if not node_ids == []:
            walks_data = embeddings.wv
            walks = []
            for node_id in node_ids:
                walks.append(walks_data[node_id])
            del walks_data

            agglomerative = AgglomerativeClustering(n_clusters=n_clusters,
                                                    affinity='cosine',
                                                    connectivity=self.Adj_M,
                                                    linkage='average'
                                                    ).fit(walks)
            return agglomerative

    def run_clustering(self, n_clusters=2, labels_dict={}, evaluate=True):
        from utilities import score_bhamidi
        from utilities import score_purity
        from utilities import score_agreement
        node_ids = []
        labels = []
        for node_id in list(labels_dict.keys()):
            node_ids.append(node_id)
            labels.append(labels_dict[node_id])
        self.kmeans = self.kmeans_evaluate(self.model,node_ids=node_ids,n_clusters=n_clusters)
        self.hierarchical = self.hierarchical_evaluate(self.model,node_ids=node_ids,n_clusters=n_clusters)
        if evaluate:
            self.bhamidi_score_hierarchical = score_bhamidi(labels, list(self.hierarchical.labels_))
            self.purity_score_hierarchical = score_purity(labels, list(self.hierarchical.labels_))
            self.agreement_score_hierarchical = score_agreement(labels, list(self.hierarchical.labels_))
            self.bhamidi_score_kmeans = score_bhamidi(labels, list(self.kmeans.labels_))
            self.purity_score_kmeans = score_purity(labels, list(self.kmeans.labels_))
            self.agreement_score_kmeans = score_agreement(labels, list(self.kmeans.labels_))
            self.kmeans_hierarchical_bhamidi = score_bhamidi(list(self.hierarchical.labels_), list(self.kmeans.labels_))
            self.kmeans_hierarchical_purity = score_purity(list(self.hierarchical.labels_), list(self.kmeans.labels_))
            self.kmeans_hierarchical_agreement = score_agreement(list(self.hierarchical.labels_), list(self.kmeans.labels_))