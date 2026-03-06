import os
import pandas as pd
import heapq
from sklearn.feature_extraction.text import CountVectorizer

CORPUS_PATH = "corpus.tsv"
OUTPUT_DIR = "."
NUM_BLOCKS = 10
DOCS_PER_BLOCK = 100
MAX_INPUT_BUFFER = 100
MAX_OUT_BUFFER = 500

# 1. Read Corpus

def read_corpus_chunks(path):
    return pd.read_csv(path, sep="\t", header=None,
                       names=["doc_id", "text"],
                       chunksize = DOCS_PER_BLOCK, encoding="utf-8")

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

# 3. Buffered multiway merge

def parse_line(line):
    line = line.rstrip("\n")

    if ":" not in line:
        return None           # safe skip
    colon = line.index(":")

    term = line[:colon]
    ids = [int(x) for x in line[colon+1:].split(",") if x]
    return term, ids

class BlockReader:
    def __init__(self, path):
        self.f = open(path, encoding="utf-8")
        self.buffer =[]
        self.exhausted = False
        self._refill()

    def _refill(self):
        while len(self.buffer) < MAX_INPUT_BUFFER and not self.exhausted:
            raw = self.f.readline()
            if not raw:
                self.exhausted = True
                break
            line = raw.rstrip("\n")
            if line and ":" in line:
                self.buffer.append(parse_line(line))

    def peek_term(self):
        return self.buffer[0][0] if self.buffer else None

    def pop(self):
        item = self.buffer.pop(0)
        #   refill when buffer gets low
        if len(self.buffer) < MAX_INPUT_BUFFER // 2:
            self._refill()
        return item

    def close(self):
        self.f.close()

def merge_blocks(block_paths, output_path):
    readers = [BlockReader(p) for p in block_paths]

    heap = []
    for i, r in enumerate(readers):
        t = r.peek_term()
        if t is not None:
            heapq.heappush(heap, (t, i))

    out_buf = []

    with open(output_path, "w", encoding="utf-8") as fout:

        def flush():
            for term, ids in out_buf:
                fout.write(f"{term}:{','.join(str(d) for d in ids)}\n")
            out_buf.clear()

        while heap:
            min_term   = heap[0][0]
            merged_ids = set()

            while heap and heap[0][0] == min_term:
                _, idx = heapq.heappop(heap)
                _, ids = readers[idx].pop()
                merged_ids.update(ids)
                next_t = readers[idx].peek_term()
                if next_t is not None:
                    heapq.heappush(heap, (next_t, idx))

            out_buf.append((min_term, sorted(merged_ids)))

            if len(out_buf) >= MAX_OUT_BUFFER:
                flush()

        flush()  # final flush — still inside the with block

    for r in readers:
        r.close()

    print(f"  Final index written to {output_path}")

