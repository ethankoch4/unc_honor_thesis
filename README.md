# unc_honor_thesis
The coding and any other work for my Honor's Thesis at UNC-Chapel Hill in Statistics and Analytics:
https://drive.google.com/drive/folders/1zCrq8eOO_JQCxtYI8K9R0BBoCYBp5uLY?ths=true


## <center>Abstract</center>
why must this be here....
:  asdlfkjalsdj asdf asd asd  omo o, om om okmomk omokmok mmo om okm om om

# 1&nbsp;&nbsp;&nbsp;&nbsp;Intro to Word2Vec

The Word2Vec algorithm was originally created by {SOURCE}. The algorithm has led to many advances in fields such as Statistics, Natural Language Processing (NLP), {HELP} due to its drastic improvement over the previous state-of-the-art methods in retaining the meaning of each word in a corpus. Benefits of the use of the Word2Vec algorithm include: the dimensionality of embeddings is significantly less than the number of documents $D$ for any reasonably sized corpus, the embeddings are dense as opposed to sparse in the case of TF-IDF, the results of the algorithm {HELP}. It is also important to note that while Word2Vec has many different architectural choices and variations, the ones focused on in the theory portion of this paper will be those that offer the core intuition underlying the algorithm and are most widely used.

## 1.1&nbsp;&nbsp;&nbsp;&nbsp;Motivation

The goal of the Word2Vec algorithm is to generate a vector for every word in a corpus that retains the meaning of that word in relation to every other word. The reason the meaning of a given word is only retained in relation to other words is that any given direction in a word's embedding, $v_j \in \mathbb{R}^s$, the direction $s_k$ itself is most likely uninterpretable in and of itself. {HELP}: https://stackoverflow.com/questions/38423387/why-does-word2vec-use-cosine-similarity

Word2Vec is an embedding algorithm with the goal of generating a vector that corresponds to a given word. This algorithm is not only the basis for many other similar embedding algorithms, but also has applications sentiment analysis, topic detection, and other NLP-related tasks. The goal of running Word2Vec on the SCOTUS corpus is to generate embeddings for the words used in Supreme Court cases for comparison with the same words used in non-legal contexts, in this case the GoogleNews embeddings {HELP}{SOURCE}.

## 1.2&nbsp;&nbsp;&nbsp;&nbsp;Brief Overview of Previous Models

### 1.2.1&nbsp;&nbsp;&nbsp;&nbsp;Bag-of-Words (BOW)

blah blah blah blah

### 1.2.2&nbsp;&nbsp;&nbsp;&nbsp;Term Frequency-Inverse Document Frequency

blah blah blah blah

## 1.2&nbsp;&nbsp;&nbsp;&nbsp;Skip-gram (SG)

The skip-gram model takes a word as its input and has a goal of predicting the words around it during training. The skip-gram model was first introduced by {SOURCE}. In an effort to illustrate how this model works, let us use the following text as an example document:

$$"Happy\ families\ are\ all\ alike;\ every\ unhappy\ family\ is\ unhappy\ in\ its\ own\ way."$$

This is the first line of *Anna Karenina* by Leo Tolstoy. Given a document of text, we generate these input-output pairs by first specifying $c$, the size of the context or window. The set of observations in the skip-gram model is then:

$$ \{(w_i,\ w_{o})\ |\ 0\leq i\leq W-1,\ -c +i\leq o \leq c+i,\ o\neq 0\} $$

Where $W$ is the number of words in our corpus. Similarly, we define $V$ as the number of *unique* words in our corpus. Now, if we specify $c$ to be 2 in our example, then some of the input-output pairs would be:

$$(Happy, families),\ (unhappy, family),$$$$(family , is),\ (families,are)$$

With this in mind, we can think about the model embedding words that appear in similar contexts near to each other. The third and fourth pairs should push the model toward embedding *is* and *are* near each other because they both appear within the context of some form of the word *family*. Mathematically, our goal is to maximize:

$$ p(w_o|w_i; \theta) = \frac{\mathcal{e}^{w_o^T\theta}}{ \displaystyle\sum_{i=0}^V \mathcal{e}^{w_i^T\theta}}$$

for a given word, $w_j$. We have seen the value we wish to maximize for a single example. However, in terms of the entirety of our corpus, we wish to find the $\theta$ which maximizes:

$$ L(\theta) = \frac{1}{V} \displaystyle\sum_{i=0}^V \displaystyle\sum_{o = -c,\ o\neq 0}^c log\ p(w_o|w_i; \theta) $$

In this setting, $w_o$ represents a vector of zeros with length $V$, where the $o^{th}$ entry is $1$.  I will also refer to it as the word it represents.

$$w_o = \begin{bmatrix}
				0 \\
				0 \\
				\vdots \\
				1 \\
				\vdots \\
				0 \\
         \end{bmatrix} $$

This is called *one-hot encoding* and serves to isolate only the row in the matrix of learned parameters, $\theta$, which correspond to that word, $w_o$. However, in practice one typically uses key/value pairs for efficiency.

The actual embedding generated, $v_o$, which corresponds to word $w_o$, is exactly the row that is isolated by multiplying our *one-hot encoded* input vector with the weight matrix:

$$v_o = w_o^T \theta$$

This is significantly different than the traditional settings where the quantities of interest are the output of a model, either predicted values or probabilities.

## 1.3&nbsp;&nbsp;&nbsp;&nbsp;Continuous Bag of Words (CBOW)

The Continuous Bag-of-Words model (CBOW) can be thought of as the reverse of the skip-gram, though it achieves the same end goal of creating embeddings for the words in a corpus. In the CBOW model, the input-output pairs are generated as follows:

$$ \{\big((w_{o-\frac{c}{2}},\dots,w_{o+\frac{c}{2}}),\ w_o\big)\ |\ \frac{c}{2}\leq o\leq W-\frac{c}{2},\  c \neq 0\} $$
One may notice in the Skip-Gram model the input-output pairs are both of the same dimension. This is not true of the CBOW model. So, we define a function $g:\ \mathbb{R}^{2c-1\times V}\to \mathbb{R}^V$ to be an element-wise averaging function (one can also define it as a concatenating function) so that on a word-level the goal becomes to maximize:

$$ p(w_o|w_{o-j},\dots,w_{o+j}; \theta) = \frac{\mathcal{e}^{\big(g(w_{o-j}^T\theta,\ \dots\ ,\ w_{o+j}^T\theta)\big)}}{ \displaystyle\sum_{i=0}^V \mathcal{e}^{\big(w_i^T\theta\big)}}$$

The CBOW model is the one we chose to run on the SCOTUS corpus for performance reasons. Which architecture of the many provides the best results is still an open question {SOURCE}.

In order to give a firm understanding of the model I have presented CBOW as using the words on either side of a given word, $w_o$, to predict $w_o$ and thereby generate embeddings. However, often times CBOW, as well as other word embeddings models, will choose $w_o$ to the word directly *after* the context. The input-output pairs are then generated in the following way:

$$ \{\big((w_{o-c},w_{o-c+1},\dots,w_{o-1}),\ w_o\big)\ |\ c\leq o\leq W\} $$

In fact, one can even choose $w_o$ to be the word directly *before* the context. However, these variations do not really alter the results of the CBOW model. They are merely preferential {SOURCE}.

These preferential choices notwithstanding, there are some architectural options to the basic SG and CBOW models I have not presented. Many of these options have the effect of making the model less computationally expensive to train and are therefore used in practice by software. Some more common options include: negative sampling, hierarchical softmax, and stochastic gradient descent. Being one of the most important model architecture options, I will provide a brief overview of stochastic gradient descent in the following section.

## 1.4&nbsp;&nbsp;&nbsp;&nbsp;Training the Model: Stochastic Gradient Descent

In Word2Vec, Doc2Vec, and Node2Vec, Stochastic Gradient Descent (SGD) is the optimization method we used to tune the parameters of the model. SGD is a form of Gradient Descent that is defined by the following steps:
>1. Choose initial parameters, typically randomly selected from a probability distribution:
$$\theta = \begin{bmatrix}
			\theta_{0,0} & \theta_{0,1} & \dots & \theta_{0,s} \\
			\theta_{1,0} & \theta_{1,1} & \dots & \theta_{1,s} \\
			\vdots & \vdots & \ddots & \vdots \\
			\theta_{V,0} & \theta_{V,1} & \dots &\theta_{V,s} \\
			         \end{bmatrix},\ \theta_{i,j}\ chosen from \  \Theta $$ 
         where $\Theta$ is some probability distribution, often $\mathcal{U}[0,1]$ or $\mathcal{N}(0,1)$. In the context of Word2Vec, $s$ is the embedding size of the word vectors, chosen beforehand, and $V$ is the number of unique words in the corpus.
>2. Calculate the gradient of the loss function over the entirety of the training data set. The parameters, $\theta$, become itself mines the calculated gradient with a learning rate.
>$$ \theta = \theta - \alpha \nabla_{\theta}L(\theta)$$
>3. Repeat step 2 until some convergence rule is achieved. This typically is a set number of iterations or when the gradient becomes sufficiently small.

Stochastic Gradient Descent is almost exactly Gradient Descent, with a small change for mostly computational purposes. It is very expensive to calculate $\nabla_{\theta}L(\theta)$, so instead we use Stochastic Gradient Descent. The difference is that step 2 is not done for every example in the training set. Instead, a subset of examples are randomly chosen from the training set when calculating the gradient of the loss function. SGD will often converge to a global minimum, and almost always converge to a local minimum, depending on the conditions {SOURCE}. SGD also usually requires more iterations than Gradient Descent for convergence, due to its use of a subset of examples.

Lastly, $\alpha$ can be set to a range of values or to decrease linearly between two values. This is effectively allowing large changes in the parameters toward the beginning of the iterations, and decreasing the change of parameters as training continues. Due to the likelihood that the parameters must change a great deal to obtain the global minimum, this approach makes sense and works in practice {SOURCE}.
# 2&nbsp;&nbsp;&nbsp;&nbsp;Intro to Doc2Vec

Word2Vec generates embeddings at a word-level. However, this is not useful if one wishes to compare, say, the abstracts of different academic articles. For this reason Doc2Vec was introduced by {SOURCE}. Doc2Vec generates embeddings for each document, $d_i \in \mathbb{R}^s$. What is considered a document is completely up to the researcher. In our case we consider each of the case opinions from SCOTUS to be a different document. Doc2Vec is almost identical to Word2Vec, with a few modifications. In fact, word embeddings are also generated as part of training a Doc2Vec model.

## 2.1&nbsp;&nbsp;&nbsp;&nbsp;Paragraph Vector - Distributed Bag of Words (PV-DBOW)

PV-DBOW is most similar to the Word2Vec Skip-Gram model. 

## 2.2&nbsp;&nbsp;&nbsp;&nbsp;Paragraph Vector - Distributed Memory (DM)

PV-DM is most similar to the Word2Vec CBOW model. Recall that in CBOW the word vectors are learned by the process of predicting 

# 3&nbsp;&nbsp;&nbsp;&nbsp;Intro to Node2Vec

blah blah blah

## 3.1&nbsp;&nbsp;&nbsp;&nbsp;Graph Object

blah blah blah

## 3.2&nbsp;&nbsp;&nbsp;&nbsp;Random Walk

blah blah blah

#### SHOULD MY PAPER BE FIRST, SECOND, OR THIRD PERSON?



# Outline of Paper:

## Introduction & Goals

## 1 - SBM Theory
***NEED TO DO***
***HOW CAN I CONNECT THIS TO Node2Vec??***
## 2 - Word2Vec, Node2Vec, Doc2Vec Theory
***NEED TO DO***
## 3 - On Real World Data
***NEED TO DO***
### 3.1 - Word2Vec on SCOTUS vs. GoogleNews vectors
***NEED TO DO***
### 3.2 - Doc2Vec vs. Node2Vec on SCOTUS (100 clusters)
***NEED TO DO***
### 3.3 - Doc2Vec vs. Node2Vec on SCOTUS (14 clusters)
***NEED TO DO***
#### 3.3.1 - Compare to issueAreas - Did either extract them?
***NEED TO DO***
#### 3.3.2 - Combine the two - Did it perform better?
***NEED TO DO***
### 3.4 - Doc2Vec vs. Node2Vec on SCOTUS (14 classifying) vs. issueAreas
***NEED TO DO***
#### 3.4.1 - Combine the two - Did it perform better?
***NEED TO DO***
### 3.5 - Overall Comparison Results of Doc2Vec vs. Node2Vec on SCOTUS
***NEED TO DO***
## 4 - Implications and Future Questions
***NEED TO DO***
