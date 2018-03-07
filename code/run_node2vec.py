def main():
    import networkx as nx
    import json
    import sys
    if len(sys.argv) < 3:
        raise ValueError("Must specify both p and q for node2vec (2nd & 3rd args after filepath")

    from utilities import load_scotus_network
    from node2vec.node2vec import node2vec

    G, issueAreas = load_scotus_network('../data/scotus_network.graphml')

    with open('../data/name_to_ia.json', 'w') as fp:
        json.dump(issueAreas, fp)
    del issueAreas
    print('P equals: {0}'.format(sys.argv[1]))
    print('Q equals: {0}'.format(sys.argv[2]))

    n2v = node2vec(G=G,
                   p=float(sys.argv[1]),
                   q=float(sys.argv[1]),
                   walk_length=150,
                   num_walks=150,
                   window_size=10,
                   embedding_size=300,
                   num_iter=100,
                   min_count=0,
                   sg=1,
                   workers=12)

    model = n2v.run_node2vec()
    model.save('../data/scotus_n2v_{0}_{1}.node2vec'.format(sys.argv[1],sys.argv[2]))

print('BEGINNING NODE2VEC SCRIPT')
main()
print('NODE2VEC SCRIPT COMPLETE')

