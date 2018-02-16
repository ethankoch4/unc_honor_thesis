def main():
    from doc2vec.doc2vec import doc2vec
    docs = ["My name is ethan and I have a lot of fun coding and doing statistics. My name is ethan and I have a lot of fun coding and doing statistics."]*20
    names = [i for i in range(len(docs))]
    Doc2vec = doc2vec(doc_list=docs,names=names)
    Doc2vec.run_doc2vec()

if __name__ == '__main__':
    main()