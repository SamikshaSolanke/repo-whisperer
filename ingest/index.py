import argparse
import uuid
from pathlib import Path
import git
from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models
from tqdm import tqdm
import config
from ingest.chunk import chunk_file
from ingest.walk import walk_repo
BATCH = 64


def get_client():
    return QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=60)


def ensure_collection(client, recreate=False):
    exists = client.collection_exists(config.COLLECTION)
    if exists and recreate:
        client.delete_collection(config.COLLECTION)
        exists = False
    if exists:
        return
    client.create_collection(
        collection_name=config.COLLECTION,
        vectors_config={"dense": models.VectorParams(
            size=config.DENSE_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams(
            modifier=models.Modifier.IDF)},
    )
    # Payload indexes make filtered search fast (repo, path, kind, language)
    for field in ("repo", "path", "kind", "language"):
        client.create_payload_index(
            config.COLLECTION, field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD)


def point_id(repo, sha, c):
    key = f"{repo}@{sha}:{c['path']}:{c['symbol']}:{c['start_line']}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. pallets/flask")
    ap.add_argument("--path", required=True, help="local clone, e.g. data/repos/flask")
    ap.add_argument("--recreate", action="store_true", help="wipe the collection first")
    args = ap.parse_args()

    root = Path(args.path)
    sha = git.Repo(root).head.commit.hexsha
    print(f"Repo {args.repo} @ {sha}")

    files = walk_repo(root)
    chunks = [c for f in files for c in chunk_file(root, f)]
    print(f"{len(files)} files -> {len(chunks)} chunks")

    client = get_client()
    ensure_collection(client, recreate=args.recreate)

    print("Loading embedding models (first run downloads them)...")
    dense = TextEmbedding(config.DENSE_MODEL)
    sparse = SparseTextEmbedding(config.SPARSE_MODEL)

    for i in tqdm(range(0, len(chunks), BATCH), desc="Indexing"):
        batch = chunks[i:i + BATCH]
        texts = [c["text"] for c in batch]
        d_vecs = list(dense.embed(texts))
        s_vecs = list(sparse.embed(texts))
        points = [
            models.PointStruct(
                id=point_id(args.repo, sha, c),
                vector={
                    "dense": d.tolist(),
                    "sparse": models.SparseVector(
                        indices=s.indices.tolist(), values=s.values.tolist()),
                },
                payload={**c, "repo": args.repo, "commit_sha": sha},
            )
            for c, d, s in zip(batch, d_vecs, s_vecs)
        ]
        client.upsert(config.COLLECTION, points=points, wait=True)

    print("Points in collection:", client.count(config.COLLECTION, exact=True).count)


if __name__ == "__main__":
    main()