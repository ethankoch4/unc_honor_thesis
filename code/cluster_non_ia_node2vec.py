def main():
    from gensim import models
    import numpy as np
    import sys

    from node2vec.node2vec import node2vec
    from utilities import load_scotus_network
    from utilities import get_name_to_date
    from utilities import get_list_of_docs
    p = sys.argv[1]
    q = sys.argv[2]
    print('n_clusters equals: {0}'.format(sys.argv[3]))

    n2v_model = node2vec(model=models.Word2Vec.load("../data/scotus_n2v_{0}_{0}_tiny.node2vec".format(p,q)))
    n2v_model.p = float(p)
    n2v_model.q = float(q)
    G, issue_areas = load_scotus_network(file_path="../data/scotus_network.graphml")

    nodes = np.random.permutation([n for n in G.nodes()])
    del G
    del issue_areas

    n2v_model.run_clustering(n_clusters=int(float(sys.argv[3])),nodes=nodes,evaluate=False)


print('BEGINNING CLUSTERING WITH NODE2VEC.')
main()
print('FINISHED CLUSTERING WITH NODE2VEC.')
