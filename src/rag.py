"""RAG layer: SEC EDGAR filings -> chunks -> Chroma (dense) + BM25 (lexical).

Ingestion:  ingest_ticker("AAPL")  — fetch recent 10-K/10-Q/8-K, chunk, embed, index.
Retrieval:  hybrid_retrieve(query, ticker, as_of_date) — dense + BM25 fused with
            Reciprocal Rank Fusion, HARD-FILTERED to filings dated BEFORE as_of_date
            (no lookahead: an explanation may only cite documents that existed
            at decision time).

Embeddings: BAAI/bge-small-en-v1.5 via sentence-transformers (CPU-friendly, ~130MB).
Store:      Chroma persistent collection in data/chroma/ ; BM25 rebuilt from the
            same collection's documents at query time.
"""
import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup

import config

# Lazy-loaded singletons
_embedder = None
_chroma_client = None

CHROMA_DIR = os.path.join(config.DATA_DIR, "chroma")
COLLECTION = "sec_filings"

SEC_USER_AGENT = config.SEC_USER_AGENT
# SEC requires a descriptive User-Agent with contact info
SEC_UA = {"User-Agent": os.getenv("SEC_USER_AGENT", SEC_USER_AGENT)}

FORMS = ("10-K", "10-Q", "8-K")
CHUNK_WORDS = 250
CHUNK_OVERLAP = 40


# --------------------------------------------------------------------------
# Embedding / storage plumbing
# --------------------------------------------------------------------------
def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return _embedder


def _get_collection():
    global _chroma_client
    import chromadb
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _chroma_client.get_or_create_collection(
        COLLECTION, metadata={"hnsw:space": "cosine"})


# --------------------------------------------------------------------------
# EDGAR ingestion
# --------------------------------------------------------------------------
def _ticker_to_cik(ticker: str) -> str:
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=SEC_UA, timeout=30)
    r.raise_for_status()
    for row in r.json().values():
        if row["ticker"].upper() == ticker.upper():
            return str(row["cik_str"]).zfill(10)
    raise ValueError(f"No CIK found for ticker {ticker}")


def _recent_filings(cik: str, max_per_form: int = 3) -> list[dict]:
    """List recent 10-K/10-Q/8-K filings from the submissions API."""
    r = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                     headers=SEC_UA, timeout=30)
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]
    out, counts = [], {f: 0 for f in FORMS}
    for form, acc, date, doc in zip(recent["form"], recent["accessionNumber"],
                                    recent["filingDate"], recent["primaryDocument"]):
        if form in FORMS and counts[form] < max_per_form:
            counts[form] += 1
            out.append({
                "form": form, "filing_date": date,
                "url": (f"https://www.sec.gov/Archives/edgar/data/"
                        f"{int(cik)}/{acc.replace('-', '')}/{doc}"),
            })
    return out


def _fetch_filing_text(url: str) -> str:
    r = requests.get(url, headers=SEC_UA, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, "lxml")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()  # tables become noise as flat text; drop for v1
    text = re.sub(r"\s+", " ", soup.get_text(" "))
    return text


def _chunk(text: str) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i: i + CHUNK_WORDS])
        if len(chunk) > 200:           # skip trivial tail fragments
            chunks.append(chunk)
        i += CHUNK_WORDS - CHUNK_OVERLAP
    return chunks


def ingest_ticker(ticker: str, max_per_form: int = 3) -> int:
    """Fetch, chunk, embed, and index recent filings. Returns chunks added.

    Idempotent per filing: already-indexed accession URLs are skipped.
    """
    ticker = ticker.upper()
    col = _get_collection()
    existing = set(col.get(where={"ticker": ticker}).get("ids", []))

    cik = _ticker_to_cik(ticker)
    added = 0
    for filing in _recent_filings(cik, max_per_form):
        base_id = filing["url"].rsplit("/", 2)[-2]          # accession number
        if any(i.startswith(f"{ticker}_{base_id}") for i in existing):
            continue                                         # already ingested
        try:
            text = _fetch_filing_text(filing["url"])
        except requests.RequestException as e:
            print(f"  skip {filing['url']}: {e}")
            continue

        chunks = _chunk(text)
        if not chunks:
            continue
        date_int = int(filing["filing_date"].replace("-", ""))  # 20260214 for $lt filters
        ids = [f"{ticker}_{base_id}_{j}" for j in range(len(chunks))]
        metas = [{"ticker": ticker, "form": filing["form"],
                  "filing_date": filing["filing_date"],
                  "filing_date_int": date_int} for _ in chunks]
        embeddings = _get_embedder().encode(chunks, show_progress_bar=False,
                                            normalize_embeddings=True).tolist()
        col.add(ids=ids, documents=chunks, metadatas=metas, embeddings=embeddings)
        added += len(chunks)
        print(f"  indexed {filing['form']} {filing['filing_date']}: {len(chunks)} chunks")
        time.sleep(0.3)                                      # be polite to SEC
    return added


# --------------------------------------------------------------------------
# Hybrid retrieval with temporal filter
# --------------------------------------------------------------------------
def _date_filter(ticker: str, as_of_date: str) -> dict:
    """Chroma where-clause: this ticker AND filing strictly before as_of_date."""
    as_of_int = int(str(as_of_date)[:10].replace("-", ""))
    return {"$and": [{"ticker": ticker.upper()},
                     {"filing_date_int": {"$lt": as_of_int}}]}


def hybrid_retrieve(query: str, ticker: str, as_of_date: str,
                    k: int = 5, candidates: int = 20) -> list[dict]:
    """Dense + BM25 retrieval fused with Reciprocal Rank Fusion (RRF).

    Both retrievers are restricted to filings dated before `as_of_date`
    ('YYYY-MM-DD') — the temporal no-lookahead guarantee.
    """
    from rank_bm25 import BM25Okapi

    col = _get_collection()
    where = _date_filter(ticker, as_of_date)

    # --- dense arm ---
    q_emb = _get_embedder().encode([query], normalize_embeddings=True).tolist()
    dense = col.query(query_embeddings=q_emb, n_results=candidates, where=where)
    dense_ids = dense["ids"][0]

    # --- BM25 arm over the same date-filtered corpus ---
    pool = col.get(where=where)
    docs, ids = pool["documents"], pool["ids"]
    if not docs:
        return []
    bm25 = BM25Okapi([d.lower().split() for d in docs])
    scores = bm25.get_scores(query.lower().split())
    bm25_ranked = [ids[i] for i in sorted(range(len(scores)),
                                          key=lambda i: -scores[i])[:candidates]]

    # --- Reciprocal Rank Fusion:  score = sum 1/(60 + rank) ---
    fused: dict[str, float] = {}
    for rank, _id in enumerate(dense_ids):
        fused[_id] = fused.get(_id, 0) + 1.0 / (60 + rank)
    for rank, _id in enumerate(bm25_ranked):
        fused[_id] = fused.get(_id, 0) + 1.0 / (60 + rank)

    top_ids = sorted(fused, key=fused.get, reverse=True)[:k]
    got = col.get(ids=top_ids)
    order = {i: n for n, i in enumerate(got["ids"])}
    results = []
    for _id in top_ids:
        n = order[_id]
        results.append({"id": _id,
                        "text": got["documents"][n],
                        "meta": got["metadatas"][n],
                        "score": fused[_id]})
    return results