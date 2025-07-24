import os
import glob
from typing import List
from dotenv import load_dotenv

from openai import AsyncOpenAI
from qdrant_client import QdrantClient, models

from openai_reranker import OpenAILlmReranker

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
RAG_DOCS_DIR = os.getenv("RAG_DOCS_DIR", "rag_docs")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

reranker = OpenAILlmReranker(api_key=OPENAI_API_KEY)

client = QdrantClient(":memory:")

collection_name = "rag_docs"

async def _embed(texts: List[str]) -> List[List[float]]:
    resp = await openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]

def _load_paragraphs() -> List[str]:
    paragraphs: List[str] = []
    for path in glob.glob(os.path.join(RAG_DOCS_DIR, "*")):
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                for p in text.split("\n\n"):
                    p = p.strip()
                    if p:
                        paragraphs.append(p)
    return paragraphs

async def init_rag() -> None:
    paragraphs = _load_paragraphs()
    if not paragraphs:
        return
    embeddings = await _embed(paragraphs)
    vector_size = len(embeddings[0])
    client.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
    )
    points = [
        models.PointStruct(id=i, vector=emb, payload={"document": doc})
        for i, (emb, doc) in enumerate(zip(embeddings, paragraphs))
    ]
    client.upsert(collection_name=collection_name, points=points)


async def query_rag(query: str, top_k: int = 3, min_score: float = 0.0) -> dict:
    query_vector = (await _embed([query]))[0]
    hits = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    docs = [hit.payload.get("document", "") for hit in hits]
    reranked = await reranker.rerank(query, docs, min_score=min_score)
    return {
        "results": [
            {"document": doc, "score": score} for doc, score in reranked
        ]
    }
