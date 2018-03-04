import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from gensim import models
# otherwise will be very slow (b.c. not using C)
assert models.doc2vec.FAST_VERSION > -1

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('fivethirtyeight')
import numpy as np

class doc2vec(object):
    def __init__(self,
            model=None,
            doc_list=[],
            names=[],
            dm=0,
            alpha=0.025,
            min_alpha=0.001,
            min_count=1,
            size=300,
            window=10,
            workers=24,
            iter=25,
            negative=10,
            sample=1e-5,
            label_docs=True,
            epochs=100,
            path_to_save='../data/scotus_model.doc2vec'
            ):
        if label_docs==True:
            self.doc_list = self.label_docs(doc_list, names)
        self.names = names
        self.dm = dm
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.min_count = min_count
        self.size = size
        self.window = window
        self.workers = workers
        self.iter = iter
        self.negative = negative
        self.sample = sample
        self.epochs = epochs
        self.path_to_save = path_to_save
        self.model = model

    def run_doc2vec(self,evaluate=True):
        model = models.Doc2Vec(
                    dm=self.dm,
                    alpha=self.alpha,
                    min_alpha=self.min_alpha,
                    min_count=self.min_count,
                    size=self.size,
                    window=self.window,
                    workers=self.workers,
                    iter=self.iter,
                    negative=self.negative,
                    sample=self.sample
                    )
        model.build_vocab(self.doc_list)

        model.train(self.doc_list,epochs=self.epochs,total_examples=model.corpus_count)
            # model.alpha *= 0.95  # decrease the learning rate
            # model.min_alpha = model.alpha  # fix the learning rate
            # print("MOST SIMILAR TO 4023639:",model.docvecs.most_similar(["4023639"])[:2])
        self.model = model
        model.save(self.path_to_save)
        return model

    def label_docs(self, doc_list, names):
        return [models.doc2vec.TaggedDocument(doc.split(),tags=[names[i]]) for i,doc in enumerate(doc_list)]

    def get_most_similar(self, doc_name, n=5, file_root='../data/scotus/textfiles/'):
        if self.model is None:
            raise AttributeError("Cannot get most similar docs of non-existent model.")
        assert isinstance(doc_name,str),"Invalid doc_name: must be str or int."

        similar_docs = [(doc_name,1.0)]
        similar_docs.extend(self.model.docvecs.most_similar(doc_name,topn=n))        

        if file_root[-1]!='/':
            file_root = file_root + '/'
        documents = []
        names = []
        for doc in iter(similar_docs):
            with open(file_root+doc[0]+".txt",'r',encoding="utf8") as readfile:
                names.append(doc[0])
                documents.append(readfile.read())
        return names, documents

    def kmeans_evaluate(self, embeddings, node_ids=[], n_clusters=2):
        from sklearn.cluster import KMeans

        if not node_ids == []:
            walks_data = embeddings.docvecs
            walks = []
            for node_id in node_ids:
                walks.append(walks_data[node_id])
            del walks_data

            kmeans = KMeans(n_clusters=n_clusters).fit(walks)
            return kmeans
            
    def hierarchical_evaluate(self, embeddings, node_ids=[], n_clusters=2):
        from sklearn.cluster import AgglomerativeClustering

        if not node_ids == []:
            walks_data = embeddings.docvecs
            walks = []
            for node_id in node_ids:
                walks.append(walks_data[node_id])
                print(walks_data[node_id])
            del walks_data

            agglomerative = AgglomerativeClustering(n_clusters=n_clusters,
                                                    affinity='cosine',
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
        print('actual labels: ',list(labels))
        print('kmeans labels: ',list(self.kmeans.labels_))
        print('hierarchical labels: ',list(self.hierarchical.labels_))
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


    def similarity_time_plot(self,case,name_to_year={},fig_size=(30,15),num_to_plot=27884,outname="ERROR",show=False):
        similar_cases = self.model.docvecs.most_similar(positive=[str(case)],topn=int(num_to_plot))
        # clip values to be between 0.0 and 1.0
        cases = [(case0[0],max(min(float(case0[1]),1.0),0.0)) for case0 in similar_cases]
        del similar_cases

        fig, ax = plt.subplots(figsize=fig_size)
        
        ax.autoscale_view()
        ax.set_title("Time vs. Similarity for ID: "+case,fontsize=24)
        ax.set_xlabel("Time (in years)",color='black',fontsize=20)
        ax.set_ylabel("Similary (between 0 and 1)",color='black',fontsize=20)
        ax.set_xlim((((min(name_to_year.values())-1)//5*5)-4,((max(name_to_year.values())+1)//5*5)+4))
        # vertical line of year corresponding to this case
        ax.axvline(name_to_year[case],c='b',label='Case of Interest')
        
        x_ticks = np.sort(np.append(np.arange((min(name_to_year.values())-1)//5*5,((max(name_to_year.values())+1)//5*5)+1,5),[name_to_year[case]]))
        ax.set_xticks(x_ticks)
        plt.setp(ax.xaxis.get_majorticklabels(),rotation=90,color='black')
        plt.setp(ax.yaxis.get_majorticklabels(),color='black')
        ax.tick_params(axis='both',color='black')

        year_to_median = {}
        x = []
        y = []
        for case0 in cases:
            # year
            year = int(name_to_year[case0[0]])
            x.append(year)
            # similarity to plotted case
            similarity = float(case0[1])
            y.append(similarity)
            # want to plot medians at each year
            if year_to_median.get(year,None) is None:
                year_to_median[year] = [similarity]
            else:
                year_to_median[year].append(similarity)
                
        medians_x = []
        medians_y = []
        for key in list(year_to_median.keys()):
            medians_x.append(key)
            medians_y.append(np.median(year_to_median[key]))

        ax.set_ylim((0.0,max(y)+0.01))
        ax.scatter(x,y,c='g',alpha=0.35,label='Cosine Similarity')
        ax.plot(medians_x,medians_y,'o-',c='r',alpha=1.0,linewidth=4,label='Median Yearly Similarity')
        legd = ax.legend(loc=0,fancybox=True,fontsize=18,scatterpoints=3)
        
        if outname != "ERROR":
            if "." not in outname[-5:]:
                fig.savefig(outname)
            else:
                fig.savefig(outname+".png")
        else:
            fig.savefig('../plots/similarity_plots/unnamed_similarity_plot.png')
        if show:
            plt.show()
        else:
            plt.close()
