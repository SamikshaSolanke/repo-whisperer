from functools import lru_cache
from fastembed import TextEmbedding
from qdrant_client import models
import config
from ingest.index import get_client


@lru_cache(maxsize=1)
def _dense():
    return TextEmbedding(config.DENSE_MODEL)


@lru_cache(maxsize=1)
def _client():
    return get_client()


def permalink(payload: dict) -> str:
    """GitHub link pinned to the indexed commit, so line numbers never drift."""
    return (f"https://github.com/{payload['repo']}/blob/{payload['commit_sha']}/"
            f"{payload['path']}#L{payload['start_line']}-L{payload['end_line']}")


def retrieve(question: str, k: int = config.TOP_K, repo: str | None = None) -> list[dict]:
    """Dense-only retrieval. Day 4 swaps this for hybrid search + reranking."""
    vec = next(iter(_dense().query_embed(question))).tolist()
    flt = None
    if repo:
        flt = models.Filter(must=[
            models.FieldCondition(key="repo", match=models.MatchValue(value=repo))])
    res = _client().query_points(
        config.COLLECTION, query=vec, using="dense",
        query_filter=flt, limit=k, with_payload=True)
    return [{**p.payload, "score": p.score, "url": permalink(p.payload)}
            for p in res.points]