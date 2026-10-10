from functools import lru_cache
from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import models
import config
from ingest.index import get_client
from query.rewrite import rewrite_query


@lru_cache(maxsize=1)
def _dense():
    return TextEmbedding(config.DENSE_MODEL)


@lru_cache(maxsize=1)
def _sparse():
    return SparseTextEmbedding(config.SPARSE_MODEL)


@lru_cache(maxsize=1)
def _reranker():
    return TextCrossEncoder(config.RERANK_MODEL)


@lru_cache(maxsize=1)
def _client():
    return get_client()


def permalink(payload: dict) -> str:
    return (f"https://github.com/{payload['repo']}/blob/{payload['commit_sha']}/"
            f"{payload['path']}#L{payload['start_line']}-L{payload['end_line']}")


def _filter(repo):
    if not repo:
        return None
    return models.Filter(must=[
        models.FieldCondition(key="repo", match=models.MatchValue(value=repo))])


def _search(query: str, repo, hybrid: bool, limit: int):
    dense_vec = next(iter(_dense().query_embed(query))).tolist()
    flt = _filter(repo)
    if not hybrid:
        res = _client().query_points(
            config.COLLECTION, query=dense_vec, using="dense",
            query_filter=flt, limit=limit, with_payload=True)
    else:
        sp = next(iter(_sparse().query_embed(query)))
        res = _client().query_points(
            config.COLLECTION,
            prefetch=[
                models.Prefetch(query=dense_vec, using="dense", filter=flt, limit=limit),
                models.Prefetch(
                    query=models.SparseVector(indices=sp.indices.tolist(),
                                              values=sp.values.tolist()),
                    using="sparse", filter=flt, limit=limit),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit, with_payload=True)
    return res.points


def _rrf(rank_lists, c: int = 60):
    """Reciprocal Rank Fusion across several ranked lists of points."""
    scores, pts = {}, {}
    for lst in rank_lists:
        for rank, p in enumerate(lst, 1):
            scores[p.id] = scores.get(p.id, 0.0) + 1.0 / (c + rank)
            pts[p.id] = p
    return [(pts[i], scores[i]) for i in sorted(scores, key=scores.get, reverse=True)]


def _penalty(payload: dict) -> float:
    parts = set(payload["path"].split("/")[:-1])
    if payload["language"] in config.DOC_LANGUAGES or parts & config.LOW_PRIORITY_DIRS:
        return config.NON_CODE_PENALTY
    return 0.0


def retrieve(question: str, k: int = config.TOP_K, repo: str | None = None,
             hybrid: bool = True, rewrite: bool = True, rerank: bool = True,
             prefer_code: bool = False) -> list[dict]:
    queries = [question] + (rewrite_query(question) if rewrite else [])
    pool = config.CANDIDATES if (rerank or len(queries) > 1) else k

    lists = [_search(q, repo, hybrid, pool) for q in queries]
    if len(lists) > 1:
        ranked = _rrf(lists)[:config.CANDIDATES]
    else:
        ranked = [(p, p.score) for p in lists[0]]

    if rerank and ranked:
        docs = [p.payload["text"] for p, _ in ranked]
        scores = list(_reranker().rerank(question, docs))
        if prefer_code:
            scores = [s - _penalty(p.payload) for (p, _), s in zip(ranked, scores)]
        ranked = sorted(zip((p for p, _ in ranked), scores),
                        key=lambda x: x[1], reverse=True)

    return [{**p.payload, "score": float(s), "url": permalink(p.payload)}
            for p, s in ranked[:k]]