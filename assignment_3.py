
import pyterrier as pt
# if not pt.started():
pt.init()


dataset = pt.datasets.get_dataset("irds:beir/trec-covid")
dataset.info


dataset.get_topics().head()