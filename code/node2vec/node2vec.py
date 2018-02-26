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
            labels=[],
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
        self.labels = labels
        self.n_classes = n_classes
        self.num_walks = num_walks
        self.walk_length = walk_length

    def run_node2vec(self):
        self.G.preprocess_transition_probs()

        self.walks = self.G.simulate_walks(self.num_walks, self.walk_length)
        
        self.model = self.learn_embeddings(self.walks)

        if self.evaluate:
            self.kmeans_evaluate(self.model,
                                labels=self.labels,
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

    def kmeans_evaluate(self, embeddings, labels=[], n_clusters=2):
        from sklearn.cluster import KMeans
        from utilities import score_bhamidi
        from utilities import score_purity
        from utilities import score_agreement

        if not labels == []:
            walks_data = embeddings.wv
            walks_data = [walks_data[str(i)] for i in range(len(labels))]

            kmeans = KMeans(n_clusters=n_clusters).fit(walks_data)
            self.bhamidi_score_kmeans = score_bhamidi(labels, list(kmeans.labels_))
            self.purity_score_kmeans = score_purity(labels, list(kmeans.labels_))
            self.agreement_score_kmeans = score_agreement(labels, list(kmeans.labels_))
            return kmeans

    def hierarchical_evaluate(self, embeddings, labels=[], n_clusters=2):
        from sklearn.cluster import AgglomerativeClustering
        from utilities import score_bhamidi
        from utilities import score_purity
        from utilities import score_agreement

        if not labels == []:
            walks_data = embeddings.wv
            walks_data = [walks_data[str(i)] for i in range(len(labels))]

            agglomerative = AgglomerativeClustering(n_clusters=n_clusters,
                                                    affinity='cosine',
                                                    connectivity=self.Adj_M,
                                                    linkage='average'
                                                    ).fit(walks_data)
            self.bhamidi_score_hierarchical = score_bhamidi(labels, list(agglomerative.labels_))
            self.purity_score_hierarchical = score_purity(labels, list(agglomerative.labels_))
            self.agreement_score_hierarchical = score_agreement(labels, list(agglomerative.labels_))
            return agglomerative
