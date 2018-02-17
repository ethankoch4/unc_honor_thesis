import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from gensim import models
# otherwise will be very slow (b.c. not using C)
assert models.word2vec.FAST_VERSION > -1

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('fivethirtyeight')
import numpy as np

class word2vec(object):
    def __init__(self,
            model=None,
            word_list=[],
            sg=0,
            alpha=0.025,
            min_alpha=0.0001,
            min_count=2,
            size=300,
            window=10,
            workers=24,
            iter=15,
            epochs=100,
            negative=10,
            sample=1e-5,
            path_to_save='../data/scotus/scotus_model.word2vec'
            ):
        self.word_list = word_list
        self.sg = sg
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.min_count = min_count
        self.size = size
        self.window = window
        self.workers = workers
        self.iter = iter
        self.epochs = epochs
        self.negative = negative
        self.sample = sample
        self.path_to_save = path_to_save
        self.model = model

    def run_word2vec(self):
        model = models.Word2vec(
                    sg=self.sg,
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
        model.build_vocab(self.word_list)

        model.train(self.word_list,epochs=self.epochs)
        self.model = model
        model.save(self.path_to_save)
        return model