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

