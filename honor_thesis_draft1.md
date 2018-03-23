---
output:
  pdf_document
#  geometry: "left=1in,right=1in,top=1in,bottom=1in"
header-includes:
  - \usepackage{setspace}\doublespacing
---

# Abstract

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* Word2Vec

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* Doc2Vec

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* Node2Vec

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* SCOTUS

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* SBM

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* Phase Transition

# 1&nbsp;&nbsp;&nbsp;&nbsp;Intro to Word2Vec, Doc2Vec, and Node2Vec

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* blah blah blah blah

## 1.1&nbsp;&nbsp;&nbsp;&nbsp;Intro to Word2Vec

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* The Word2Vec algorithm was originally created by [{SOURCE]}. The algorithm has led to many advances in fields such as Statistics, Natural Language Processing (NLP), [{HELP]} due to its drastic improvement over the previous state-of-the-art methods in retaining the meaning of each word in a corpus. Benefits of the use of the Word2Vec algorithm include: the dimensionality of embeddings is significantly less than the number of documents $D$ for any reasonably sized corpus, the embeddings are dense as opposed to sparse in the case of TF-IDF, the results of the algorithm [{HELP}]. It is also important to note that while Word2Vec has many different architectural choices and variations, the ones focused on in the theory portion of this paper will be those that offer the core intuition underlying the algorithm and are most widely used.

### 1.1.1&nbsp;&nbsp;&nbsp;&nbsp;Motivation

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* The goal of the Word2Vec algorithm is to generate a vector for every word in a corpus that retains the meaning of that word in relation to every other word. The reason the meaning of a given word is only retained in relation to other words is that any given direction in a word's embedding, $w_iv_j \in \mathbb{R}^ks$, the direction $k_js_k$ itself is most likely uninterpretable in and of itself. [{HELP}: https://stackoverflow.com/questions/38423387/why-does-word2vec-use-cosine-similarity]

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* Word2Vec is an embedding algorithm with the goal of generating a vector that corresponds to a given word. This algorithm is not only the basis for many other similar embedding algorithms, but also has applications sentiment analysis, topic detection, and other NLP-related tasks. The goal of running Word2Vec on the SCOTUS corpus is to generate embeddings for the words used in Supreme Court cases for comparison with the same words used in non-legal contexts, in this case the GoogleNews embeddings {HELP}{SOURCE}.

### 1.1.2&nbsp;&nbsp;&nbsp;&nbsp;Skip-gram Architecture

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* blah blah blah blah

## 1.2&nbsp;&nbsp;&nbsp;&nbsp;Brief Overview of Previous Models

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* blah blah blah blah

### 1.2.1&nbsp;&nbsp;&nbsp;&nbsp;Bag-of-Words (BOW)

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* blah blah blah blah

### 1.2.2&nbsp;&nbsp;&nbsp;&nbsp;Term Frequency-Inverse Document Frequency

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* blah blah blah blah

## 1.2&nbsp;&nbsp;&nbsp;&nbsp;Skip-gram (SG)

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* The skip-gram model takes a word as its input and has a goal of predicting the words around it during training. The skip-gram model was first introduced by {SOURCE}. In an effort to illustrate how this model works, let us use the following text as an example document:

$$ Happy\ families\ are\ all\ alike;\ every\ unhappy\ family\ is\ unhappy\ in\ its\ own\ way $$

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* This is the first line of *Anna Karenina* by Leo Tolstoy. Given a document of text, we generate these input-output pairs by first specifying $c$, the size of the context or window. The set of observations in the skip-gram model is then:

$$ \big\{(w_oi,\ w_{o-j})\ \big|\ 0\leq oi\leq W-1,\ -c +i\leq jo \leq c+i,\ jo\neq 0\big\} $$

where $W$ is the number of words in our corpus. Similarly, we define $V$ as the number of *unique* words in our corpus. Now, if we specify $c$ to be 2 in our example, then some of the input-output pairs would be:

$$ (Happy,\ families),\ (unhappy,\ family), $$
$$ (family,\ is),\ (families,\ are) $$

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* With this in mind, we can think about the model embedding words that appear in similar contexts near to each other. The third and fourth pairs should push the model toward embedding *is* and *are* near each other because they both appear within the context of some form of the word *family*. Mathematically, our goal is to maximize:

$$ p(w_o\big|w_i; \theta) = \frac{e^{\big(w_0^T\theta\big)}}{\displaystyle\sum_{i=0}^V e^{\big(w_i^T\theta\big)}} $$

for a given word, $w_oj$. We have seen the value we wish to maximize for a single example. However, in terms of the entirety of our corpus, we wish to find the $\theta$ which maximizes:

$$ L(\theta) = \frac{1}{V} \displaystyle\sum_{i=0}^V \displaystyle\sum_{o = -c,\ o\neq 0}^c log\ p(w_o\big|w_i; \theta) $$

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* In this setting, $w_o$ represents a vector of zeros with length $V$, where the $o^{th}$ entry is $1$.  I will also refer to it as the word it represents.

$$w_o = \begin{bmatrix}
				0 \\
				0 \\
				\vdots \\
				1 \\
				\vdots \\
				0 \\
\end{bmatrix} $$

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* This is called *one-hot encoding* and serves to isolate only the row in the matrix of learned parameters, $\theta$, which correspond to that word, $w_o$. 

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* We have seen the value we wish to maximize for a single example. However, in terms of the entirety of our corpus, we wish to find the $\theta$ whichHowever, in practice one typically uses key/value pairs for efficiency.

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* The actual embedding generated, $v_o$, which corresponds to word $w_o$, is exactly the row that is isolated by multiplying our *one-hot encoded* input vector with the weight matrix:

$$ v_o = w_o^T \theta $$

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* This is significantly different than the traditional settings where the quantities of interest are the output of a model, either predicted values or probabilities.

## 1.3&nbsp;&nbsp;&nbsp;&nbsp;Continuous Bag of Words (CBOW)

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* The Continuous Bag-of-Words model (CBOW) can be thought of as the reverse of the skip-gram, though it achieves the same end goal of creating embeddings for the words in a corpus. In the CBOW model, the input-output pairs are generated as follows:

$$ \big\{\big((w_{o-\frac{c}{2}},\dots,w_{o+\frac{c}{2}}),\ w_o\big)\ \big|\ \frac{c}{2}\leq o\leq W-\frac{c}{2},\  c \neq 0\big\} $$

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* In the context of our *Anna Karenina* example, with $c = 2$, this corresponds to a few input-output pairs being:

$$ \big((Happy,\ are),\ families\big),\ \big((all,\ every),\ alike\big),$$
$$ \big((every,\ family),\ unhappy\big),\ \big((its,\ way),\ own\big) $$

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* One may notice that the input-output pairs are not of the same dimension in CBOW, though the algebra in training will require this to be true. So, we define a function $g:\ \mathbb{R}^{2c-1\times V}\to \mathbb{R}^V$ to be an element-wise averaging function (one can also define it as a concatenating function) so that on a word-level the goal becomes to maximizes:

$$ p(w_o\big|w_{o-j},\dots,w_{o+j}; \theta) = \frac{e^{\big(g(w_{o-j}^T\theta,\ \dots\ ,\ w_{o+j}^T\theta)\big)}}{ \displaystyle\sum_{i=0}^V e^{\big(w_i^T\theta\big)}} $$

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* The CBOW model is the one we chose to run on the SCOTUS corpus for performance reasons. Which architecture of the many provides the best results is still an open question {SOURCE}.

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* In order to give a firm understanding of the model I have presented CBOW as using the words on either side of a given word, $w_o$, to predict $w_o$ and thereby generate embeddings. However, one can alter CBOW, as well as other word embeddings models, to define $w_o$ as the word directly *after* the context. The input-output pairs are then generated in the following way:

$$ \big\{\big((w_{o-c},w_{o-c+1},\dots,w_{o-1}),\ w_o\big)\ \big|\ c\leq o\leq W\big\} $$



*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* In fact, one can even choose $w_o$ to be the word directly *before* the context. However, these variations do not really alter the results of the CBOW model. They are merely preferential {SOURCE}.

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* These preferential choices notwithstanding, there are some architectural options to the basic SG and CBOW models I have not presented. Many of these options have the effect of making the model less computationally expensive to train and are therefore used in practice by software. Some more common options include: negative sampling, hierarchical softmax, and stochastic gradient descent. Being one of the most important model architecture options, I will provide a brief overview of stochastic gradient descent in the following section.

## 1.4&nbsp;&nbsp;&nbsp;&nbsp;Training the Model: Stochastic Gradient Descent

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* In Word2Vec, Doc2Vec, and Node2Vec, Stochastic Gradient Descent (SGD) is the optimization method we used to tune the parameters of the model. SGD is a form of Gradient Descent that is defined by the following steps:

> 1. Choose initial parameters, typically randomly selected from a probability distribution:
$$\theta = \begin{bmatrix}
			\theta_{0,0} & \theta_{0,1} & \dots & \theta_{0,s} \\
			\theta_{1,0} & \theta_{1,1} & \dots & \theta_{1,s} \\
			\vdots & \vdots & \ddots & \vdots \\
			\theta_{V,0} & \theta_{V,1} & \dots &\theta_{V,s} \\
\end{bmatrix},\ \theta_{i,j}\ chosen from \  \Theta $$ 
where $\Theta$ is some probability distribution, often $\mathcal{U}[0,1]$ or $\mathcal{N}(0,1)$. In the context of Word2Vec, $s$ is the embedding size of the word vectors, chosen beforehand, and $V$ is the number of unique words in the corpus.
> 2. Calculate the gradient of the loss function over the entirety of the training data set. The parameters, $\theta$, become itself mines the calculated gradient with a learning rate.
> $$ \theta = \theta - \alpha \nabla_{\theta}L(\theta)$$
> 3. Repeat step 2 until some convergence rule is achieved. This typically is a set number of iterations or when the gradient becomes sufficiently small.


*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* Stochastic Gradient Descent is almost exactly Gradient Descent, with a small change for mostly computational purposes. It is very expensive to calculate $\nabla_{\theta}L(\theta)$, so instead we use Stochastic Gradient Descent. The difference is that step 2 is not done for every example in the training set. Instead, a subset of examples are randomly chosen from the training set when calculating the gradient of the loss function. SGD will often converge to a global minimum, and almost always converge to a local minimum, depending on the conditions {SOURCE}. SGD also usually requires more iterations than Gradient Descent for convergence, due to its use of a subset of examples.

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* Lastly, $\alpha$ can be set to a range of values or to decrease linearly between two values. This is effectively allowing large changes in the parameters toward the beginning of the iterations, and decreasing the change of parameters as training continues. Due to the likelihood that the parameters must change a great deal to obtain the global minimum, this approach makes sense and works in practice {SOURCE}.

# 2&nbsp;&nbsp;&nbsp;&nbsp;Intro to Doc2Vec

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* Word2Vec generates embeddings at a word-level. However, this is not useful if one wishes to compare, say, the abstracts of different academic articles. For this reason Doc2Vec was introduced by {SOURCE}. Doc2Vec generates embeddings for each document, $d_i \in \mathbb{R}^s$. What is considered a document is completely up to the researcher. One convenient aspect of Doc2Vec is that the documents can be of variable legnth. In our case we consider each of the case opinions from SCOTUS to be a different document. Doc2Vec is almost identical to Word2Vec, with a few modifications. In fact, word embeddings are also generated as part of training a Doc2Vec model.

## 2.1&nbsp;&nbsp;&nbsp;&nbsp;Paragraph Vector - Distributed Bag of Words (PV-DBOW)

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* PV-DBOW is most similar to the Word2Vec Skip-Gram model. The vectors that correspond to a given document here are learned by a process of predicting the words in $d_i$. The input-ouput pairs are generated as follows:

$$ \big\{\big(d_i,\ (w_{d_{i,j}},\dots,w_{d_{i,j+c}})\big)\ \big|\ 0\leq i\leq s,\ 0\leq j \leq W_{d_i}-c\big\} $$

where $s$ is the number of documents, $W_{d_i}$ is the length of the sequence of words corresponding to document $d_i$, $w_{d_{i,j}}$ is the $j^{th}$ word in the sequence of words corresponding to document $d_i$, and $c$ is the size of the context or window.

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* To further illustrate this let us extend the *Anna Karenina* example we have begun. Consider the corpus to be the first line in *Anna Karenina* by Leo Tolstoy and the last line in *The Great Gatsby* by F. Scott Fitzgerald:

$$ (So\ we\ beat\ on,\ boats\ against\ the\ current,\ borne\ back\ ceaselessly\ into\ the\ past, $$
$$ Happy\ families\ are\ all\ alike;\ every\ unhappy\ family\ is\ unhappy\ in\ its\ own\ way) $$

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* Then, giving the ids $0$, and $1$, respectively, a few of the input-output pairs, with $c = 2$ would be:

$$ (0,\ boats\ against),\ (0,\ ceaselessly\ into), $$
$$ (1,\ Happy\ families),\ (1,\ unhappy\ family) $$

where we note that the ids and words would be replaced with one-hot encoded vectors in the actual algorithm.

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* The PV-DBOW method, according the original paper, only needs to "store the softmax weights as opposed to both softmax weights and word vectors" as in the model we will discuss next, PV-DM, resulting in PV-DBOW to utilize less memory in training {SOURCE}. The final output of the model is a vector corresponding to each document in the corpus. Ideally, documents that contain similar sequences of words will be mapped near each other in the resulting vector space. In these algorithms, cosine similarity is used as the measure of how "similar" two vectors are. Henceforth, cosine similarity will be what we mean when we talk about two vectors being similar.

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* Because this is an unsupervised algorithm, one must do further analysis to identify how well the algorithm has worked on the corpus in question. Generally, PV-DBOW works better in practice, so this is the algorithm we chose to run on the SCOTUS corpus {SOURCE: https://arxiv.org/pdf/1607.05368.pdf}.

## 2.2&nbsp;&nbsp;&nbsp;&nbsp;Paragraph Vector - Distributed Memory (DM)

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* PV-DM is most similar to the Word2Vec CBOW model. Recall that in CBOW the word vectors are learned by the process of predicting a specific word given a context. The extension to making this a document-embedding algorithm is quite straightforward. All one must do is include the document id to the input as if it were an additional context word. Thus, the input-output pairs are generated as follows:

$$ \big\{\big((d_i,\ w_{i,o-\frac{c}{2}},\dots,w_{i,o+\frac{c}{2}}),\ w_o\big)\ \big|\ \frac{c}{2}\leq o\leq W_{d_i}-\frac{c}{2},\ c \neq 0\big\} $$

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* In the context of our *Anna Karenina* and *The Great Gatsby* example, this would result in a few of the input-output pairs as follows:

$$ \big((0,\ borne,\ ceaselessly),\ back\big),\ \big((0,\ on,\ against),\ boats\big), $$
$$ \big((1,\ Happy,\ are),\ families\big),\ \big((1,\ every,\ family),\ unhappy\big) $$
with $c = 2$ and the words and ids corresponding to one-hot encoded vectors, of course.

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* Though the extension to Doc2Vec is straightforward, it turns out to be a very powerful algorithm as evidenced by {SOURCE}. This algorithm also results in word embeddings due to the words being used as inputs along with the document id. This does cause the PV-DM algorithm to be more memory-intensive than its counterpart, but allows one to avoid also training a Word2Vec model separately, if both are intended to be used. Because we opted to use PV-DBOW on the SCOTUS corpus, we ran Doc2Vec and Word2Vec separately.

# 3&nbsp;&nbsp;&nbsp;&nbsp;Intro to Node2Vec

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* Another extension of Word2Vec is Node2Vec, an embedding algorithm for graph or network data sets. Whereas Word2Vec and Doc2Vec are algorithms that were inspired by creating embeddings given sequences of words, the actual algorithm is agnostic to the word itself. The only importance a word provides to the algorithm is the vector to which it is associated. Thus, the Word2Vec and Doc2Vec models could just as easily use sequencesfof numbers, letters, or any combination of the two to output embeddings. Node2Vec leverages this by generating a sequence of node ids which will later be used in the Word2Vec algorithm in order to generate embeddings for the nodes in a graph. The key insight of this alogrithm is the method by which these sequences of node ids are generated. At the end of the algorithm, similar nodes should be near each other.

## 3.1&nbsp;&nbsp;&nbsp;&nbsp;Graph Object

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* A graph or network, $G$, is defined as an ordered pair of its nodes, $V$, and edges which connect the nodes, $E$, so that $G = (V, E)$. We only consider undirected graphs in this paper. To make this and the following ideas concrete, let us consider a simple graph of cities, where an undirected edge exists if one can drive from one city to the other one a single highway.
$$ V = (Charlotte,\ Atlanta,\ Nashville,\ Birmingham) $$
$$ E = \big((Charlotte,\ Atlanta),\ (Atlanta,\ Birmingham),\ (Nashville,\ Birmingham) \big) $$

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* The adjacency matrix, $A \in \mathbb{R}^{|V| \times |V|}$, is used to mathematically represent this graph structure. The entry, $A_{i,j}$, is $1$ if there exists an edge between node $i$ and $j$; otherwise, the entry is $0$. Note that the adjacency matrix is symmetric about its diagonal entries. The adjacency matrix for our example looks like:

$$ A = \begin{bmatrix}
			 0 & 1 & 0 & 0 \\
			 1 & 0 & 0 & 1 \\
			 0 & 0 & 0 & 1 \\
			 0 & 1 & 1 & 0 \\
			 \end{bmatrix} $$

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* The *homophily* hypothesis states that nodes that are connected are similar and therefore should me near each other after embeddings are generated {SOURCE}. In our small example, this means that after running an embedding algorithm we would hope to find that: 1) Charlotte is similar to Atlanta, but not to Birmingham and Nashville, 2) Atlanta is similar to Charlotte and Birmingham, but not to Nashville, 3) Nashville is similar to Birmingham, but not to Atlanta and Charlotte, and 4) Birmingham is similar to Atlanta and Nashville, but not to Charlotte.

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* In contrast to *homophily* is *structural equivalence*. The hypothesis of *structural equivalence* is that similar nodes play similar roles in the network. In context, of our example, it would mean we would hope to see Atlanta and Birmingham similar to each other, because they are both the link connecting two different cities. Additionally, Charlotte and Nashville should be similar, because they are both connected only to the linking nodes, and no others.

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* There are trade-offs that one must make when choosing between these two sampling approaches. Breadth-First approaches consider all nodes immediately connected to the current node when choosing which node to visit next. Depth-First approaches consider nodes in increasing distance from the sources when choosing the next node. Breadth-First approaches favor the homophily hypothesis, whereas Depth-First approaches favor the structural equivalence hypothesis. That is, when generating a sequence of nodes, if one believed the homophily hypothesis to better encapsulate the relationships between the graph's nodes, one would employ the BFS approach to generate a random sequence. This would result in nodes that are connected often being immediately next or near each other in the sequence. In our example, a few random walks favoring this approach may look as follows:

$$ (Charlotte,\ Atlanta,\ Charlotte,\ Atlanta,\ Birmingham) $$
$$ (Atlanta,\ Birmingham,\ Atlanta,\ Birmingham,\ Nashville) $$

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* On the other hand, if one believed that the relationship between nodes more closely resembled the structural equivalence hypothesis, it would be better to favor the DFS approach, in which the structure of the graph is more easily apparent in the sequence. A few random walks favoring this approach may look as follows:

$$ (Charlotte, Atlanta, Birmingham, Nashville, Birmingham) $$
$$ (Nashville, Birmingham, Atlanta, Charlotte, Atlanta) $$

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* The reason that algorithms using graph structures must choose between these approaches is that there is no natural ordering of the data like in text data, where every sentence can be taken as a sequence, or time-series data, where the progression of time provides natural, sequential ordering to the data. Additionally, there is generally no *start* or *end* to the graphs like in the given examples of other data types.

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* Most often in real-world applications, however, some nodes are better described by homophily and others structural equivalence. An exclusively BFS or DFS approach will not accomodate the differences between nodes. The *Random Walk* from the Node2Vec algorithm addresses this issue.

## 3.2&nbsp;&nbsp;&nbsp;&nbsp;Random Walk

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* The random walk in this algorithm is, as was previously stated, the key contribution of this paper. It is the method by which a sequence of nodes is generated. This algorithm creates a single walk by starting a given node, $V_i$, and using a probability distribution over the previously visited node, the current node, and the immediate neighbors of the current node, chooses a "next" node in the sequence, to be repeated until the walk is of the desired length. A node is considered a neighbor of another node if they are connected by an edge. This is repeated a specific number of times for each node in the graph. Thus, the parameters specific to the random walk algorithm are as follows:

> $p$ : Return parameter, a higher value (larger than $max(q,1)$) favors a BF approach, $p \in \mathbb{R}_{\geq 0}$
>
> $q$ : In-out parameter, a higher value (larger than 1) favors a DF approach, $q \in \mathbb{R}_{\geq 0}$
>
> $m$ : Number of random walks to generate per node, $m \in \mathbb{N}^+$
>
> $l$ : Number of nodes sampled in each random walk, $m \in \mathbb{N}^+$


*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* One may notice that this will result in a matrix, $W \in V^{m\cdot|V|\ \times\ l}$, with the rows corresponding to random walks, each of length $l$. In our previous example, if we choose $m = 1$ and $l = 4$, we may obtain the following matrix:

$$ W = \begin{bmatrix}
			 Charlotte & Atlanta & Birmingham & Nashville  \\
			 Atlanta & Birmingham & Nashville & Birmingham  \\
			 Birmingham & Atlanta & Charlotte & Atlanta  \\
			 Nashville & Birmingham & Atlanta & Charlotte  \\
			 \end{bmatrix} $$

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* After the walks are generated, the examples are ready to be run with the Word2Vec algorithm in order to obtain node embeddings. However, let us first take a closer look at how the "next" node is chosen at each step in a given walk.

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* The probability distribution used to model the next possible nodes is {SOURCE}:

$$ p(c_i = x \big| c_{i-1} = v) = \begin{cases}
\pi_{v,x}, & \text{if}\ (v,\ x) \in E \\
0, & \text{otherwise}
\end{cases} $$

where the constant, $Z$ is used to normalize the unnormalized probabilities, $\pi_{v,x}$, that guide the walk, so that they sum to $1$. Now, let $\pi_{v,x} = \alpha_{p,q}(t,x) \cdot A_{v,x}$, and these unnormailized probabilities are obtained as follows:

$$ \alpha_{p,q}(t,x) = \begin{cases}
\frac{1}{p}, & \text{if}\ d_{t,x} = 0 \\
1, & \text{if}\ d_{t,x} = 1 \\
\frac{1}{q}, & \text{if}\ d_{t,x} = 2 \\
\end{cases} $$

where $d_{t,x}$ is the shortest path between nodes $t$ and $x$ {SOURCE}.

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* This value, $d_{t,x}$, will be exactly $0$ for only the node $t$, exaclty $1$ if $x$ is a neighbor of $t$, and exactly $2$ if $x$ is a neighbor of $v$ and not $x$. Thus, it is clear why $p$ is considered the return parameter, as it dictates the probability with which a node will return to node the walk has just visited. Likewise, $q$ is considered the in-out parameter because it dictates the probability with which the walk will move further from previously visited nodes, thereby getting "out" of the current node neighborhood, into new parts of the graph.

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* In practice, it is much too expensive to calculate these probability distributions while generating the random walks, so they are calculated prior to generating walks for each possible scenario. This is more memory-intensive, but drastically decreases the amount of time required to generate the random walks, because often the same task is repeated multiple times, given that each node begins a random walk $m$ number of times.

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* Less work has been done with Node2Vec than Word2Vec or Doc2Vec, though it is a promising node embedding algorithm and has many practical applications ranging from Law, to Social Networks, to Medicine, and more. Some of the work that has been done with Word2Vec is how to go about choosing the hyperparameters of the algorithm {SOURCE}. This has yet to be done with Node2Vec and the following portion of my paper begins to deal with this.

# 4&nbsp;&nbsp;&nbsp;&nbsp;Stochastic Block Model

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* While it is tricky to identify how a specific algorithm performs as a specific parameter varies on a real, messy data set, one may perform such a task on a synthetic data set where the ground truth is known. For example, identifying how well Node2Vec performs on the SCOTUS citation network is not helpful because not only are we not aware of the upper bound of the results of the algorithm, but there is also variability in the existence of edges that may or may not be due to the node itself, we cannot know. Thus, we use the Stochastic Block Model (SBM) to run simulations looking at how well the algorithm is able to perform under certain parameter settings.

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* The SBM is a random graph that is created by defining a probability matrix, $B \in \mathbb{R}^{|C| \times |C|}$, where $|C|$ is the number of communities decided beforehand and the entry, $B_{i,j}$ is the probability that an edge will exist for two given nodes, one in community $i$ and community $j$. In the simplest setting, one defines the probability of connecting to nodes within the same community and the probability of connecting to nodes outside of a node's community. All information is known about a graph generated from an SBM because one need only define the number of nodes in each community, and then for every node in a given community the edges will be created according to the specified probability matrix. Note that neither the rows, nor the columns, in the probability matrix are required to sum to $1$, as is required in a probability distribution.

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* Due to the existence of a few parameters in the SBM itself, I chose to hold a few constant (e.g. number of nodes in each community, probability of connecting to a node within of one's community), and vary the parameter corresponding to the probability of connecting to a node outside of a given node's community while also varying a Node2Vec parameter. This is quite computationally expensive due to the number of simulations required, so while some progress has been made, it is not a complete picture, because that would potentially take years to run on current computer systems. The results of this will be discussed in a later section.

# Intro to SCOTUS Corpus

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* The Supreme Court of the United States releases an opinion for every case that they hear. Our data set from CourtListener has 27,885 of these {SOURCE}.   Thus, viewing the SCOTUS corpus as a graph, this data set has 27,885 nodes (corresponding to each case) and 234,209 edges, where an edge exists if one case cites another. In my case I consider the graph as being undirected. My hope is that this will remove the time series aspect of the data that is difficult to account for. Additionally, this simplifies the implementation of previously discussed algorithms. This citation graph is precisely the data set on which I ran Node2Vec to generate embeddings for each case.

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* Each case (node) has text associated with it as well. I did not attempt to account for differences in author of the case, though I do not believe this would affect the embeddings greatly, since the cases are generally quite thorough and discuss the topic of the case at length, as opposed to using colloquial language more specific to a single judge. I did no further preprocessing, though I do believe very slight improvements may be made using methods such as tokenization due to language variations over time. In fact, attempting to account for language variations over time would be an interesting extension. However, it is outside the scope of this project. It is on these text documents that I ran Doc2Vec to generate embeddings for each case.

# Further Work

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* While work has been done on Doc2Vec regarding finding the optimal parameters given a certain setting, no work has been done of this type for Node2Vec. Thus, my next task was to begin looking at the change in how well the algorithm performed as a specific parameter varied. -------->

# 5&nbsp;&nbsp;&nbsp;&nbsp;On Real World Data

## 5.1&nbsp;&nbsp;&nbsp;&nbsp;Word2Vec on SCOTUS vs. GoogleNews vectors

***NEED TO DO***

## 5.2&nbsp;&nbsp;&nbsp;&nbsp;Doc2Vec Similarity Plots

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* After running the Doc2Vec algorithm on the SCOTUS corpus, I became curious about how similarity is affected by time. So, for each document I calculated its similarity to every other document and plotted the similarity over time. In the following plot the blue line corresponds to the date of the document held constant, each green dot corresponds to the similarity score between the document held constant and some other document in the corpus, and the red line is the median similarity score at each year.

![](2645639_similarity_plot.png)

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* As seen in the above plot, every case after about 2006 displays a marked jump in similarity scores between pre-2006 cases and post-2006. The reason for this is not yet clear and further investigation must be done to identify the cause of this jump, but one initial thought is that certain procedural precedents may have changed that are referred to in every post-2006 case. However, even this is not likely, since this has certainly happened in the court's past and jumps like this are not seen in previous cases. As an example, the plot for this case from 1892 is much more stable over time:

![](2539855_similarity_plot.png)

*&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* *&nbsp;* Thus, while this jump seems to contain interesting differences and may merit further analysis, it will require the recruitment of legal experts and further investigation of the cases themselves instead of the algorithms performed on them, which is outside of the scope of this paper. Let's move on to comparing the Doc2Vec and Node2Vec algorithms run on the SCOTUS corpus.

## 5.3&nbsp;&nbsp;&nbsp;&nbsp;Doc2Vec vs. Node2Vec on SCOTUS: Varying Cluster Sizes

![](hierarchical_d2v_n2v_first.png)

![](kmeans_d2v_n2v_first.png)

![](n2v_clusters_first.png)
![](d2v_clusters_first.png)

## 5.4&nbsp;&nbsp;&nbsp;&nbsp;Doc2Vec vs. Node2Vec on SCOTUS vs. IssueAreas

***NEED TO DO***

### 5.4.1&nbsp;&nbsp;&nbsp;&nbsp;Combine the two - Did it perform better?

***NEED TO DO***

## 5.5&nbsp;&nbsp;&nbsp;&nbsp;Phase Transition on SBM

![](emp_thresh_vs_wl.png)

![](redo_q_0.8_walk_len_50.png)



# 5.5&nbsp;&nbsp;&nbsp;&nbsp;Conclusion

***NEED TO DO***
