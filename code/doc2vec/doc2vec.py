from gensim import models
# otherwise will be very slow (b.c. not using C)
assert models.doc2vec.FAST_VERSION > -1

class doc2vec(object):
    def __init__(self,
            doc_list=[],
            names=[],
            dm=0,
            alpha=0.01,
            min_alpha=0.001,
            min_count=2,
            size=300,
            window=10,
            workers=24,
            iter=15,
            negative=10,
            sample=1e-5,
            label_docs=True,
            epochs=100,
            path_to_save='../data/scotus/scotus_model.doc2vec'
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

    def run_doc2vec(self):
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

        model.save(self.path_to_save)
        return model

    def label_docs(self, doc_list, names):
        return [models.doc2vec.LabeledSentence(doc.split(),tags=[names[i]]) for i,doc in enumerate(doc_list)]