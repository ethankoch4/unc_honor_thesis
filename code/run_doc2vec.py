def main():
    from gensim import models

    from doc2vec.doc2vec import doc2vec
    from utilities import load_scotus_network
    from utilities import get_name_to_date
    from utilities import get_list_of_docs

    # so I don't accidentally start training again
    NOT_YET_TRAINED = True
    if NOT_YET_TRAINED:
        doc_list, names = get_list_of_docs(dir_path='../data/local/scotus/textfiles/*.txt')
        Doc2vec = doc2vec(doc_list=doc_list,names=names)
        Doc2vec.run_doc2vec()
        del Doc2vec

print('ABOUT TO TRAIN DOC2VEC.')
main()
print('DONE TRAINING DOC2VEC.')
