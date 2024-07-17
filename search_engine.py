"""
Simple indexer and search engine built on an inverted-index and the BM25 ranking algorithm.
"""
import os
from collections import defaultdict, Counter
import pickle
import math
import operator
import nltk
import code

from tqdm import tqdm
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from datasets import load_dataset

# please un-comment the below line of you are running it first time
# nltk.download('wordnet')


class Indexer:
    dbfile = "./ir.idx"                 # file to save/load the index

    def __init__(self):
        self.tok2idx = {}
        self.idx2tok = {}
        self.postings_lists = defaultdict(list)
        self.docs = []
        self.raw_ds = None
        self.corpus_stats = {'avgdl': 0}
        self.stopwords = set(stopwords.words('english'))
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.lemmatizer = WordNetLemmatizer()

        if os.path.exists(self.dbfile):         # verify if the index file exists
            with open(self.dbfile, 'rb') as f:
                self.tok2idx, self.idx2tok, self.postings_lists, self.docs, self.raw_ds, self.corpus_stats = pickle.load(f)
        else:
            ds = load_dataset("cnn_dailymail", '3.0.0', split="test")
            self.raw_ds = ds['article']
            self.clean_text(self.raw_ds)
            self.create_postings_lists()
            with open(self.dbfile, 'wb') as f:
                pickle.dump((self.tok2idx, self.idx2tok, self.postings_lists, self.docs, self.raw_ds, self.corpus_stats), f)

    def clean_text(self, lst_text, query=False):
        cleaned_docs = []
        for doc in tqdm(lst_text, desc="cleaning the text: "):
            tokens = self.tokenizer.tokenize(doc.lower())
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stopwords]
            cleaned_docs.append(tokens)
        if query:                                           # If request is coming frm a query
            return cleaned_docs[0]
        else:                                               # else a list of documents are being processed
            self.docs = cleaned_docs

    def create_postings_lists(self):
        doc_lens = []
        for doc_id, doc in tqdm(enumerate(self.docs), desc="Creating postings: "):
            doc_lens.append(len(doc))
            term_freq = Counter(doc)
            for term, freq in term_freq.items():
                if term not in self.tok2idx:
                    self.tok2idx[term] = len(self.tok2idx)
                    self.idx2tok[self.tok2idx[term]] = term
                term_id = self.tok2idx[term]
                self.postings_lists[term_id].append((doc_id, freq))

        self.corpus_stats['avgdl'] = sum(doc_lens) / len(doc_lens)
        self.corpus_stats['doc_count'] = len(self.docs)
        self.corpus_stats['doc_lens'] = doc_lens


class SearchAgent:
    k1 = 1.5            # BM25 parameter k1 for tf saturation
    b = 0.75            # BM25 parameter b for document length normalization

    def __init__(self, indexer):
        self.i = indexer            # indexer object for further use

    def query(self, q_str):
        query_tokens = self.i.clean_text([q_str], query=True)
        doc_scores = defaultdict(float)
        N = self.i.corpus_stats['doc_count']
        avgdl = self.i.corpus_stats['avgdl']

        for term in tqdm(query_tokens, desc="Calculating scores: "):
            if term in self.i.tok2idx:
                term_id = self.i.tok2idx[term]
                df = len(self.i.postings_lists[term_id])            # Calculates the document frequency
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)     # Calculates the IDF
                for doc_id, freq in self.i.postings_lists[term_id]:
                    doc_len = self.i.corpus_stats['doc_lens'][doc_id]
                    # Calculates the BM25 score for the term in the current document
                    score = idf * (freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * (doc_len / avgdl)))
                    doc_scores[doc_id] += score

        sorted_scores = sorted(doc_scores.items(), key=operator.itemgetter(1), reverse=True)
        self.display_results(sorted_scores)

    def display_results(self, results):
        for docid, score in tqdm(results[:5], desc="printing results: "):  # print top 5 results
            print(f'\nDocID: {docid}')
            print(f'Score: {score}')
            print('Article:')
            print(self.i.raw_ds[docid])


if __name__ == "__main__":
    i = Indexer()                                           # instantiate an indexer
    q = SearchAgent(i)                                      # document retriever
    code.interact(local=dict(globals(), **locals()))        # interactive shell
