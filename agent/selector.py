"""SELECT stage- chunk fetched pages, embed+rank vs user query, pick best subset.

Picks within 3 constraints: max_snippets, token_budget, max_per_domain.
Domain cap forces source diversity- also surfaces conflict between sources.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import numpy as np
import tiktoken
from sentence_transformers import SentenceTransformer

from config import MAX_CONTEXT_TOKENS, MAX_SNIPPETS_PER_TURN, SNIPPET_CHAR_LIMIT
from tools.fetch import FetchedPage


@dataclass
class Snippet:
    text: str
    url: str
    title: str
    domain: str
    score: float
    publish_date: str | None = None   #propagated from FetchedPage, used by recency boost


#LAZY SINGLETONS- load once per process (first call pays the cost)
_model: SentenceTransformer | None = None
_tokenizer = None


def _embedder() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")  #80MB, 384-dim, fast on CPU
    return _model


def _tok():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.get_encoding("cl100k_base")  #gpt-4 tokenizer, ~10% off llama but close enough for budgeting
    return _tokenizer


def count_tokens(text: str) -> int:
    return len(_tok().encode(text))


#CHUNKING- paragraph first, sentence fallback for long paragraphs
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"“‘])")


def _chunk_text(text: str, limit: int = SNIPPET_CHAR_LIMIT) -> list[str]:
    """Paragraph-first, sentence-fallback chunking.
    - split on blank lines
    - drop paragraphs <100 chars (boilerplate noise that survived trafilatura)
    - long paragraphs split at sentence boundary and re-packed up to `limit`
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    for p in paras:
        if len(p) < 100:
            continue
        if len(p) <= limit:
            chunks.append(p)
            continue
        #pack sentences greedily up to limit
        sents = _SENT_SPLIT.split(p)
        cur = ""
        for s in sents:
            if len(cur) + len(s) + 1 <= limit:
                cur = (cur + " " + s).strip() if cur else s
            else:
                if cur:
                    chunks.append(cur)
                cur = s
        if cur:
            chunks.append(cur)
    return chunks


def _all_snippets_from_pages(pages: Iterable[FetchedPage]) -> list[Snippet]:
    out: list[Snippet] = []
    for p in pages:
        if not p.text or p.error:   #skip failed/empty pages
            continue
        for ch in _chunk_text(p.text):
            out.append(Snippet(text=ch, url=p.url, title=p.title,
                               domain=p.domain, score=0.0,
                               publish_date=p.publish_date))
    return out


#RECENCY FACTOR- gentle multiplier on similarity score
#returns 1.0 (today) decaying to 0.3 (floor) for very old/no-date content
def _recency_factor(publish_date_str: str | None) -> float:
    if not publish_date_str:
        return 0.5   #neutral when date is unknown (many pages lack metadata)
    try:
        dt = datetime.fromisoformat(publish_date_str[:10])
        days_old = (datetime.now() - dt).days
        if days_old < 0:
            return 1.0
        #linear decay: today=1.0, 730 days=0.0, floor at 0.3
        return max(0.3, 1.0 - days_old / 730)
    except Exception:
        return 0.5


#RANKING + BUDGETED SELECTION
def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return b_norm @ a_norm


def score_snippets(pages: list[FetchedPage], user_query: str) -> list[Snippet]:
    """Every snippet from every page, scored and ranked. No caps applied.

    Split out of select() so a second batch of pages can be scored on the same
    scale and merged with what a first batch produced. The scores are
    comparable across calls because nothing here depends on the rest of the
    batch: each snippet is scored against the query alone.
    """
    snippets = _all_snippets_from_pages(pages)
    if not snippets:
        return []

    #EMBED query + all snippets, score by cosine similarity
    model = _embedder()
    q_emb = model.encode([user_query], normalize_embeddings=False)[0]
    s_emb = model.encode([s.text for s in snippets], normalize_embeddings=False,
                         batch_size=32, show_progress_bar=False)
    scores = _cosine(q_emb, s_emb)

    #COMBINE relevance + recency: score = cosine * (0.7 + 0.3 * recency)
    #effect: today's content keeps full cosine; 1yr old loses ~15%; 2yr old or no-date ~21%
    #gentle enough not to override strong relevance matches, strong enough to break ties toward fresh
    for s, sc in zip(snippets, scores):
        recency = _recency_factor(s.publish_date)
        s.score = float(sc) * (0.7 + 0.3 * recency)

    snippets.sort(key=lambda s: s.score, reverse=True)   #rank descending
    return snippets


def pack(snippets: list[Snippet],
         max_snippets: int = MAX_SNIPPETS_PER_TURN,
         token_budget: int = MAX_CONTEXT_TOKENS,
         max_per_domain: int = 2) -> list[Snippet]:
    """Greedy walk down a ranked list, honouring the domain cap and the budget.

    The pattern that matters: break on max_snippets, because the quota is full
    and nothing further down can change that; continue on the cap or the
    budget, because a smaller snippet further down may still fit.

    Sorts defensively rather than trusting the caller, since a caller that
    merged two ranked lists is holding one that is no longer ranked.
    """
    ordered = sorted(snippets, key=lambda s: s.score, reverse=True)
    chosen: list[Snippet] = []
    per_domain: dict[str, int] = {}
    used_tokens = 0
    for s in ordered:
        if len(chosen) >= max_snippets:
            break
        if per_domain.get(s.domain, 0) >= max_per_domain:
            continue
        cost = count_tokens(s.text)
        if used_tokens + cost > token_budget:
            continue
        chosen.append(s)
        per_domain[s.domain] = per_domain.get(s.domain, 0) + 1
        used_tokens += cost
    return chosen


def select(pages: list[FetchedPage], user_query: str,
           max_snippets: int = MAX_SNIPPETS_PER_TURN,
           token_budget: int = MAX_CONTEXT_TOKENS,
           max_per_domain: int = 2) -> list[Snippet]:
    """Rank snippets by similarity to user query, walk top-down picking
    within token_budget and max_per_domain cap."""
    return pack(score_snippets(pages, user_query),
                max_snippets, token_budget, max_per_domain)
