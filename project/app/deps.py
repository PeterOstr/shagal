from mailkb.infra import get_clickhouse_client, get_qdrant_client, get_embeddings


def check_connections() -> dict:
    result = {
        "clickhouse": False,
        "qdrant": False,
        "embeddings": False,
    }

    try:
        ch = get_clickhouse_client()
        ch.command("SELECT 1")
        result["clickhouse"] = True
    except Exception:
        result["clickhouse"] = False

    try:
        qdrant = get_qdrant_client()
        qdrant.get_collections()
        result["qdrant"] = True
    except Exception:
        result["qdrant"] = False

    try:
        emb = get_embeddings()
        emb.embed_query("test")
        result["embeddings"] = True
    except Exception:
        result["embeddings"] = False

    return result