# Initialing ===================================================================
import pyterrier as pt

if not pt.started():
    pt.init()

# Loading TREC dataset =========================================================
dataset = pt.datasets.get_dataset("irds:beir/trec-covid")
# dataset.info
dataset.get_topics().head()

# Indexing using pyterrier =====================================================
import os

pt_index_path = './indices/cord19'

if not os.path.exists(pt_index_path + "/data.properties"):
    indexer = pt.index.IterDictIndexer(pt_index_path, blocks=True)
    index_ref = indexer.index(dataset.get_corpus_iter(),
                              fields=['title', 'text'],  # Ensure these fields exist in the dataset
                              meta=('docno',))

else:
    index_ref = pt.IndexRef.of(pt_index_path + "/data.properties")

index = pt.IndexFactory.of(index_ref)

# Runnnig BM25 Retriever =======================================================
topics = dataset.get_topics('query')
qrels = dataset.get_qrels()

# bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", controls={"c": 0.75, "bm25.k_1": 0.75, "bm25.k_3": 0.75})
bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25", controls={"c": 0.3, "bm25.k_1": 1.2, "bm25.k_3": 20})
TF_IDF = pt.BatchRetrieve(index_ref, wmodel="TF_IDF")
PL2 = pt.BatchRetrieve(index_ref, wmodel="PL2")

pipe = bm25 >> (TF_IDF ** PL2)

res = pipe.transform(topics)
print(f"BM25: {bm25.getControl('c')}, {bm25.getControl('bm25.k_1')}, {bm25.getControl('bm25.k_3')}")

# Evaluating ===================================================================
qrels = dataset.get_qrels()

eval_metrics = ['P_10', 'ndcg_cut_10', 'map']
exp_res = pt.Experiment(
    [bm25],
    topics,
    qrels,
    eval_metrics=eval_metrics,
)
exp_res
print(f"baseline: {exp_res}")

# Assignment Starts from here ==================================================
# Re-ranking with MonoT5 =======================================================
import pyterrier_t5
from pyterrier_t5 import MonoT5ReRanker

# Load the MonoT5 re-ranker
monoT5 = pyterrier_t5.MonoT5ReRanker()

# Apply BM25 and then re-rank with MonoT5
pipeline = bm25 >> pt.text.get_text(dataset, "text") >> monoT5

exp_res_reranked = pt.Experiment(
    [pipeline],
    topics,
    qrels,
    eval_metrics=eval_metrics,
)

print("reranked: ", exp_res_reranked)

