def main():
    from gensim import models
    import numpy as np

    from doc2vec.doc2vec import doc2vec
    from utilities import load_scotus_network
    from utilities import get_name_to_date
    from utilities import get_list_of_docs

    d2v_model = doc2vec(model=models.Doc2Vec.load("../data/scotus_model.doc2vec"),label_docs=False)

    G, issue_areas = load_scotus_network(file_path="../data/scotus_network.graphml")

    IA = 15
    
    nodes = np.random.permutation([n for n in G.nodes])

    ia_to_name = {i : [] for i in range(IA)}
    name_to_ia = {}
    for n in nodes:
        ia = int(float(G.nodes[n]['issueArea']))
        ia_to_name[ia].append(n)
        name_to_ia[n] = ia

    total = 0
    for k in list(ia_to_name.keys()):
        print('Key : ',k," "*(3-len(str(k))),'Length(list at key): ',len(ia_to_name[k]))
        total += len(ia_to_name[k])
        
    print('Total: ',total)

    print('Number of keys of name_to_ia: ',len(name_to_ia.keys()))
    print('The above two numbers should be equal.')

    d2v_model.run_clustering(n_clusters=len(set(ia_to_name.keys())),labels_dict=name_to_ia,evaluate=True)


print('BEGINNING CLUSTERING WITH DOC2VEC.')
main()
print('FINISHED CLUSTERING WITH DOC2VEC.')