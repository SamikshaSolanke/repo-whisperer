from fastembed import TextEmbedding
from ingest.index import get_client
import config

client = get_client()
dense = TextEmbedding(config.DENSE_MODEL)
q = "How does Flask register a blueprint on an application?"
vec = next(iter(dense.query_embed(q))).tolist()

res = client.query_points(config.COLLECTION, query=vec, using="dense", limit=5)
for p in res.points:
    pl = p.payload
    print(f"{p.score:.3f}  {pl['path']}:{pl['start_line']}-{pl['end_line']}  {pl['symbol']}")