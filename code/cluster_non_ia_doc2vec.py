def main():
    from gensim import models
    import numpy as np
    import sys

    from doc2vec.doc2vec import doc2vec
    from utilities import load_scotus_network
    from utilities import get_name_to_date
    from utilities import get_list_of_docs

    print('n_clusters equals: {0}'.format(sys.argv[1]))

    d2v_model = doc2vec(model=models.Doc2Vec.load("../data/scotus_model.doc2vec"),label_docs=False)

    G, issue_areas = load_scotus_network(file_path="../data/scotus_network.graphml")

    nodes = np.random.permutation([n for n in G.nodes])
    del G
    del issue_areas

    d2v_model.run_clustering(n_clusters=int(float(sys.argv[1])),nodes=nodes,evaluate=False)


print('BEGINNING CLUSTERING WITH DOC2VEC.')
main()
print('FINISHED CLUSTERING WITH DOC2VEC.')