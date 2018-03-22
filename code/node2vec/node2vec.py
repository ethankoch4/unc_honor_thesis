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
            model=None
            ):
        if model is None:
            if G:
                nx_G = G
            else:
                nx_G = self.read_graph(Adj_M)
            self.G = Node2VecGraph(nx_G, False, p, q)
            
            if Adj_M is None:
                self.Adj_M = nx.to_numpy_matrix(nx_G)
            else:
                self.Adj_M = Adj_M
        else:
            self.model = model
        self.evaluate = evaluate
        self.n_classes = n_classes
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = p
        self.q = q

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
                                                    linkage='average'
                                                    ).fit(walks)
            return agglomerative

    def run_clustering(self, n_clusters=2, labels_dict={}, nodes=[], evaluate=True):
        from utilities import score_bhamidi
        from utilities import score_purity
        from utilities import score_agreement
        from utilities import save_current_status
        node_ids = []
        labels = []
        if labels_dict!={}:
            for node_id in list(labels_dict.keys()):
                node_ids.append(node_id)
                labels.append(int(labels_dict[node_id]))
        else:
            node_ids = nodes
        print('Clustering now.')
        node_ids = [str(_id) for _id in node_ids]
        labels = [int(lab) for lab in labels]
        self.kmeans = self.kmeans_evaluate(self.model,node_ids=node_ids,n_clusters=n_clusters)
        self.hierarchical = self.hierarchical_evaluate(self.model,node_ids=node_ids,n_clusters=n_clusters)
        self.kmeans_labels = [int(lab) for lab in self.kmeans.labels_]
        self.hierarchical_labels = [int(lab) for lab in self.hierarchical.labels_]
        file_name = "ia_node2vec_clustering_{0}_{1}".format(self.p,self.q)
        if labels_dict == {}:
            file_name = str(n_clusters) + '_non_' + file_name
        print('Finished clustering. Saving data now.')
        save_current_status(file_name=file_name,
                            n_clusters=n_clusters,
                            true_labels=labels,
                            node_ids=node_ids,
                            kmeans_labels=self.kmeans_labels,
                            hierarchical_labels=self.hierarchical_labels,
                            )
        # save most recent model
        dir_path = 'data/'
        kmeans_file = 'ia_recent_kmeans_n2v_{0}_{1}.pickle'.format(self.p,self.q)
        hierarchical_file = 'ia_recent_hierarchical_n2v_{0}_{1}.pickle'.format(self.p,self.q)
        if labels_dict=={}:
            kmeans_file = str(n_clusters) + '_non_' + kmeans_file
            hierarchical_file = str(n_clusters) + '_non_' + hierarchical_file
        # make sure you find 'data/' directory
        if os.path.isdir(dir_path):
            kmeans_file = dir_path + kmeans_file
            hierarchical_file = dir_path + hierarchical_file
        elif os.path.isdir('../' + dir_path):
            kmeans_file = '../' + dir_path + kmeans_file
            hierarchical_file = '../' + dir_path + hierarchical_file
        elif os.path.isdir('../' + '../' + dir_path):
            kmeans_file = '../' + '../' + dir_path + kmeans_file
            hierarchical_file = '../' + '../' + dir_path + hierarchical_file
        else:
            kmeans_file = '../' + dir_path + kmeans_file
            hierarchical_file = '../' + dir_path + hierarchical_file
        # make folders if dont exist
        os.makedirs(os.path.dirname(kmeans_file), exist_ok=True)
        os.makedirs(os.path.dirname(hierarchical_file), exist_ok=True)
        pickle.dump(self.kmeans, open(kmeans_file, 'wb'))
        pickle.dump(self.hierarchical, open(hierarchical_file, 'wb'))
        if evaluate:
            print('Finished saving. Evaluating now.')
            self.bhamidi_score_hierarchical = score_bhamidi(labels, self.hierarchical_labels)
            print('Finished: score_bhamidi(labels, self.hierarchical_labels)')
            self.purity_score_hierarchical = score_purity(labels, self.hierarchical_labels)
            print('Finished: score_purity(labels, self.hierarchical_labels)')
            self.agreement_score_hierarchical = score_agreement(labels, self.hierarchical_labels)
            print('Finished: score_agreement(labels, self.hierarchical_labels)')
            self.bhamidi_score_kmeans = score_bhamidi(labels, self.kmeans_labels)
            print('Finished: score_bhamidi(labels, self.kmeans_labels)')
            self.purity_score_kmeans = score_purity(labels, self.kmeans_labels)
            print('Finished: score_purity(labels, self.kmeans_labels)')
            self.agreement_score_kmeans = score_agreement(labels, self.kmeans_labels)
            print('Finished: score_agreement(labels, self.kmeans_labels)')
            self.kmeans_hierarchical_bhamidi = score_bhamidi(self.hierarchical_labels, self.kmeans_labels)
            print('Finished: score_bhamidi(self.hierarchical_labels, self.kmeans_labels)')
            self.kmeans_hierarchical_purity = score_purity(self.hierarchical_labels, self.kmeans_labels)
            print('Finished: score_purity(self.hierarchical_labels, self.kmeans_labels)')
            self.kmeans_hierarchical_agreement = score_agreement(self.hierarchical_labels, self.kmeans_labels)
            print('Finished: score_agreement(self.hierarchical_labels, self.kmeans_labels)')
            print('Finished evaluating. Saving now, then exiting.')
            save_current_status(file_name="node2vec_clustering_evaluated_{0}_{1}".format(self.p,self.q),
                                n_clusters=n_clusters,
                                true_labels=labels,
                                node_ids=node_ids,
                                kmeans_labels=self.kmeans_labels,
                                hierarchical_labels=self.hierarchical_labels,
                                bhamidi_score_hierarchical=self.bhamidi_score_hierarchical,
                                purity_score_hierarchical=self.purity_score_hierarchical,
                                agreement_score_hierarchical=self.agreement_score_hierarchical,
                                bhamidi_score_kmeans=self.bhamidi_score_kmeans,
                                purity_score_kmeans=self.purity_score_kmeans,
                                agreement_score_kmeans=self.agreement_score_kmeans,
                                kmeans_hierarchical_bhamidi=self.kmeans_hierarchical_bhamidi,
                                kmeans_hierarchical_purity=self.kmeans_hierarchical_purity,
                                kmeans_hierarchical_agreement=self.kmeans_hierarchical_agreement
                                )
            print('Exiting.')
