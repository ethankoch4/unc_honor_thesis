usepackage{amssymb}
    





### <center>Abstract</center>
why must this be here....
:  asdlfkjalsdj asdf asd asd  omo o, om om okmomk omokmok mmo om okm om om

## 1    Intro to Word2Vec

The Word2Vec algorithm was originally created by [{SOURCE]}. The algorithm has led to many advances in fields such as Statistics, Natural Language Processing (NLP), [{HELP]} due to its drastic improvement over the previous state-of-the-art methods in retaining the meaning of each word in a corpus. Benefits of the use of the Word2Vec algorithm include: the dimensionality of embeddings is significantly less than the number of documents $D$ for any reasonably sized corpus, the embeddings are dense as opposed to sparse in the case of TF-IDF, the results of the algorithm [{HELP].
 }. It is also important to note that while Word2Vec has many different architectural choices and variations, the ones focused on in the theory portion of this paper will be those that offer the core intuition underlying the algorithm and are most widely used.

### 1.1    Motivation

The goal of the Word2Vec algorithm is to generate a vector for every word in a corpus that retains the meaning of that word in relation to every other word. The reason the meaning of a given word is only retained in relation to other words is that any given direction in a word's embedding, $w_i \in \mathbb{R}^k$, the direction $k_j$ itself is most likely uninterpretable in and of itself. [{HELP: https://stackoverflow.com/questions/38423387/why-does-word2vec-use-cosine-similarity]

Word2Vec is an embedding algorithm with the goal of generating a vector that corresponds to a given word

 Doc2Vec, and Node2Vec } This algorithm is not only the basis for many other similar embedding algorithms, but also has applications sentiment analysis, topic detection, and other NLP-related tasks. The goal of running Word2Vec on the SCOTUS corpus is to generate embeddings for the words used in Supreme Court cases for comparison with the same words used in non-legal contexts, in this case the GoogleNews embeddings {HELP}{SOURCE}.

### 1.2    Skip-gram Architecture

The skip-gram model takes a word as its input and has a goal of predicting the words around it during training. The skip-gram model was first introduced by {SOURCE}. In an effort to illustrate how this model works, let us use the following text as an example document:

$$"Happy\ families\ are\ all\ alike;\ every\ unhappy\ family\ is\ unhappy\ in\ its\ own\ way."$$

This is the first line of *Anna Karenina* by Leo Tolstoy. Given a document of text, we generate these input-output pairs by first specifying $c$, the size of the context or window. The set of observations in the skip-gram model is then:

$$ \{(w_o,\ w_{o-j})\ |\ 0\leq o\leq W,\ -c\leq j \leq c,\ j\neq 0\} $$

Where $W$ is the number of words in our corpus. Similarly, we define $V$ as the number of *unique* words in our corpus. Now, if we specify $c$ to be 2 in our example, then some of the input-output pairs would be:

$$(Happy, families), (unhappy, family),$$$$(family , is), (families,are)$$

With this in mind, we can think about the model embedding words that appear in similar contexts near to each other. The third and fourth pairs should push the model toward embedding *is* and *are* near each other because they both appear within the context of some form of the word *family*. Mathematically, our goal is to maximize:

$$ p(w_o|w_i; \theta) = \frac{exp(w_o^T\theta)}{ \displaystyle\sum_{i=0}^V exp(w_i^T\theta)}$$

for a given word, $w_o$. We have seen the value we wish to maximize for a single example. However, in terms of the entirety of our corpus, we wish to find the $\theta$ which maximizes:

$$ \frac{1}{V} \displaystyle\sum_{i=0}^V \displaystyle\sum_{j = -c,\ j\neq 0}^c log\ p(w_o|w_i; \theta) $$


In this setting, $w_o$ represents a vector of zeros with length $V$, where the $o^{th}$ entry is $1$.  I will also refer to it as the word it represents.

$$w_o = \begin{bmatrix}
				0 \\
				\vdots \\
				1 \\
				\vdots \\
				0 \\
         \end{bmatrix} $$

This is called *one-hot encoding* and serves to isolate only the row in the matrix of learned parameters, $\theta$, which correspond to that word, $w_o$.

The actual embedding generated, $v_o$, which corresponds to word $w_o$, is exactly the row that is isolated by multiplying our *one-hot encoded* input vector with the weight matrix:

$$v_o = w_o^T \theta$$

This is significantly different than the traditional settings where the quantities of interest are the output of a model, either predicted values or probabilities.

{HELP} TALK ABOUT LOGISTIC REGRESSION, SOFTMAX, ETC.
### 1.3    Continuous Bag-of-Words Architecture

The Continuous Bag-of-Words model (CBOW) can be thought of as the reverse of the skip-gram, though it achieves the same end goal of creating embeddings for the words in a corpus. In the CBOW model, the input-output pairs are generated as follows:

$$ \{((w_{o-j},\ w_o)\ |\ 0\leq o\leq V,\ -c\leq j \leq c,\ j\neq 0\} $$

The CBOW model is the one we chose to run on the SCOTUS corpus for performance reasons. In general, the two different architectures provide very similar results.

### 1.4 

## 2    Intro to Doc2Vec

Word2Vec generates embeddings at a word-level. However, this is not useful if one wishes to compare, say, the abstracts of d
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTQxMDEzMjgwNl19
-->