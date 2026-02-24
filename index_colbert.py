import colbert
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert.data import Queries, Collection
import pickle
from datasets import load_dataset

if __name__ == "__main__":
    checkpoint = 'colbert-ir/colbertv1.9'
    nbits = 2  # encode each dimension with 2 bits
    doc_maxlen = 512  # truncate passages at 300 tokens

    index_name = "medcorp"
    with open('/data/data_user_alpha/MedRAG/corpus/medcorp_chunks.pickle', 'rb') as file:
        collection = pickle.load(file)

    with Run().context(RunConfig(nranks=4, experiment='knowledge')):  # nranks specifies the number of GPUs to use
        config = ColBERTConfig(doc_maxlen=doc_maxlen, nbits=nbits) # kmeans_niters specifies the number of iterations of k-means clustering; 4 is a good and fast default.
                                                                                    # Consider larger numbers for small datasets.
        indexer = Indexer(checkpoint=checkpoint, config=config)
        indexer.index(name=index_name, collection=collection, overwrite='resume')
