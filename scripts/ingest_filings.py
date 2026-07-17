"""Ingestion pipeline CLI: fetch + index SEC filings for a ticker.

    python scripts/ingest_filings.py AAPL
    python scripts/ingest_filings.py AAPL --max-per-form 4
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import rag

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("ticker")
    p.add_argument("--max-per-form", type=int, default=3)
    args = p.parse_args()

    print(f"Ingesting SEC filings for {args.ticker.upper()}...")
    n = rag.ingest_ticker(args.ticker.upper(), args.max_per_form)
    print(f"Done — {n} new chunks indexed.")