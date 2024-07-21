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

eval_metrics=['P_10', 'ndcg_cut_10', 'map']
exp_res = pt.Experiment(
    [bm25],
    topics,
    qrels,
    eval_metrics=eval_metrics,
)
exp_res


# Assignment Starts from here ==================================================
# Re-ranking with MonoT5 =======================================================
import pyterrier_t5
from transformers import T5Tokenizer, T5ForConditionalGeneration

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

print(exp_res_reranked)