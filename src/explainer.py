"""Explainer agent: combines RL decision + ML forecast + sentiment with
retrieved SEC filing context and produces a grounded natural-language summary
via a Hugging Face Inference API LLM.

Design rule: the LLM EXPLAINS the already-made decision. It never decides,
never alters numbers, and must cite which filing supports each claim.
"""
import os
from datetime import date

from huggingface_hub import InferenceClient

import config
from src import rag

HF_TOKEN = config.HF_TOKEN
SEC_USER_AGENT = config.SEC_USER_AGENT
LLM_MODEL = os.getenv("HF_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

SYSTEM_PROMPT = """You are a financial analysis assistant embedded in a \
decision-support system. You are given (a) STRUCTURED SIGNALS produced by \
quantitative models and (b) RETRIEVED CONTEXT from SEC filings dated before \
the decision date.

Rules:
- The trading decision is already made. Explain it; do not second-guess or \
change it.
- Never state a number that is not present in the structured signals or the \
retrieved context.
- Cite the filing (form + date) for every claim drawn from context, e.g. \
[10-Q 2026-05-02].
- If the retrieved context does not support a relevant claim, say \
"not addressed in retrieved filings" rather than inventing one.
- 5-8 sentences, plain professional English."""


def _format_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return "(no filings available before the decision date)"
    lines = []
    for i, c in enumerate(chunks, 1):
        m = c["meta"]
        lines.append(f"[chunk {i}: {m['form']} {m['filing_date']}] "
                     f"{c['text'][:700]}")
    return "\n\n".join(lines)


def explain(ticker: str, decision_action: str, confidence: float,
            ml_forecast: float, sentiment: float,
            as_of_date: str | None = None, k: int = 4) -> dict:
    """Retrieve context and generate the combined summary.

    Returns {"summary": str, "sources": [chunk dicts], "model": str}.
    """
    as_of = as_of_date or date.today().isoformat()

    # Retrieval query built from the situation, not hardcoded per ticker
    query = (f"{ticker} outlook revenue guidance risks margin demand "
             f"{'growth catalysts' if decision_action == 'BUY' else 'risk factors headwinds'}")
    chunks = rag.hybrid_retrieve(query, ticker, as_of, k=k)

    user_prompt = f"""STRUCTURED SIGNALS (authoritative):
- Ticker: {ticker}
- Decision date: {as_of}
- Final decision: {decision_action} (RL policy confidence {confidence:.0%})
- ML ensemble forecast, next-day log return: {ml_forecast:+.4f}
- FinBERT daily news sentiment score: {sentiment:+.2f}  (range -1 to +1)

RETRIEVED CONTEXT (SEC filings before {as_of}):
{_format_chunks(chunks)}

Task: Write the combined analysis summary following the rules."""

    token = os.getenv("HF_TOKEN", HF_TOKEN)
    if not token:
        return {"summary": ("HF_TOKEN not set — add it to your .env or "
                            "Codespaces secrets to enable the LLM summary."),
                "sources": chunks, "model": None}

    client = InferenceClient(model=LLM_MODEL, token=token)
    resp = client.chat_completion(
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_prompt}],
        max_tokens=400,
        temperature=0.2,
    )
    summary = resp.choices[0].message.content.strip()
    return {"summary": summary, "sources": chunks, "model": LLM_MODEL}