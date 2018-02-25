def main():
      import networkx as nx

      from utilities import load_scotus_network
      from node2vec.node2vec import node2vec

      G, issueAreas = load_scotus_network('../data/scotus_network.graphml')

      with open('../data/name_to_it.json', 'w') as fp:
          json.dump(issueAreas, fp)
      del issueAreas

      n2v = node2vec(G=G,
                     p=1,
                     q=1,
                     walk_length=650,
                     num_walks=650,
                     window_size=10,
                     embedding_size=300,
                     num_iter=100,
                     min_count=0,
                     sg=1,
                     workers=24)

      model = n2v.run_node2vec()

      model.save('../data/scotus_n2v')

print('BEGINNING NODE2VEC SCRIPT')
main()
print('NODE2VEC SCRIPT COMPLETE')

