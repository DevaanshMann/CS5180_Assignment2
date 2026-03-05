import os
import heapq
from sklearn.feature_extraction.text import CountVectorizer

CORPUS_PATH = "corpus.tsv"
OUTPUT_DIR = "."
NUM_BLOCKS = 10
DOCS_PER_BLOCK = 100
MAX_INPUT_BUFFER = 100
MAX_OUT_BUFFER = 500

# 1. Read Corpus

def read_corpus(path):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t",1)
            if len(parts) == 2:
                docs.append((parts[0], parts[1]))

    # sort by numeric part: D0001 -> 1
    docs.sort(key = lambda x: int(x[0][1:]))
    return docs

# 2. SPIMI block construction

def build_block(doc_batch):
    vectorizer = CountVectorizer(stop_words="english", binary = True)
    texts = [text for _, text in doc_batch]
    doc_ids = [int(doc_id[1:]) for doc_id, _ in doc_batch]

    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return {}

    terms = vectorizer.get_feature_names_out()
    index = {}

    cx = X.tocsc()

    for col_idx, term in enumerate(terms):
        rows = cx.getcol(col_idx).nonzero()[0]
        posting = sorted(doc_ids[r] for r in rows)
        if posting:
            index[term] = posting

    return index

def write_block(index, block_num):
    path = os.path.join(OUTPUT_DIR, f"block_{block_num}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for term in sorted(index):
            posting_str = ",".join(str(d) for d in index[term])
            f.write(f"{term}:{posting_str}\n")
    print(f"Wrote {path} ({len(index)} terms)")
    return path

