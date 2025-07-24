from __future__ import annotations

import json
import re
from typing import List, Tuple

from openai import AsyncOpenAI

class OpenAILlmReranker:
    """Rerank documents for a query using an OpenAI chat model."""

    def __init__(self, api_key: str, model: str = "gpt-4o") -> None:
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def rerank(
        self, query: str, docs: List[str], min_score: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Return (document, score) pairs ordered by score and filtered."""
        if not docs:
            return []

        doc_list = "\n".join(f"{i + 1}. {doc}" for i, doc in enumerate(docs))
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that scores each document from 0 to 10 "
                    "by its relevance to the user query. Respond with a JSON array of "
                    "objects containing 'index' and 'score', sorted by score."
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nDocuments:\n{doc_list}\n\nReturn the scores.",
            },
        ]
        resp = await self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=0
        )
        content = resp.choices[0].message.content
        try:
            items = json.loads(content)
        except json.JSONDecodeError:
            pattern = r"(\d+)\D+(\d+(?:\.\d+)?)"
            items = [
                {"index": int(i), "score": float(s)} for i, s in re.findall(pattern, content)
            ]

        scored_docs = [
            (docs[item["index"] - 1], float(item["score"]))
            for item in items
            if 1 <= item["index"] <= len(docs)
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score in scored_docs if score >= min_score]
