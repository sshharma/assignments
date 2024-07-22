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
bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
topics = dataset.get_topics('query')

res = bm25.transform(topics)
res

# Evaluating ===================================================================
qrels = dataset.get_qrels()

eval_metrics = ['P_10', 'ndcg_cut_10', 'map']
exp_res = pt.Experiment(
    [bm25],
    topics,
    qrels,
    eval_metrics=eval_metrics,
)
print(exp_res)

# Adding RM3 for Query Expansion ================================================
# Apply RM3 for query expansion
bm25_rm3 = bm25 >> pt.rewrite.RM3(index)

exp_res_rm3 = pt.Experiment(
    [bm25_rm3],
    topics,
    qrels,
    eval_metrics=eval_metrics,
)
print(exp_res_rm3)

# Assignment Starts from here ==================================================
# Re-ranking with MonoT5 =======================================================
import pyterrier_t5
from pyterrier_t5 import MonoT5ReRanker

# Load the MonoT5 re-ranker
monoT5 = pyterrier_t5.MonoT5ReRanker()

# Apply BM25 + RM3 and then re-rank with MonoT5
pipeline_rm3 = bm25_rm3 >> pt.text.get_text(dataset, "text") >> monoT5

exp_res_reranked_rm3 = pt.Experiment(
    [pipeline_rm3],
    topics,
    qrels,
    eval_metrics=eval_metrics,
)

print(exp_res_reranked_rm3)
