import re
from typing import List, Tuple

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam


class OpenAILlmReranker:
    """Rerank documents for a query using an OpenAI chat model."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def rerank(
        self, query: str, docs: List[str], min_score: int = 5
    ) -> List[Tuple[str, int]]:
        """Return (document, score) pairs ordered by score and filtered."""

        if not docs:
            return []

        # Formatear contexto como en DEFAULT_CHOICE_SELECT_PROMPT_TMPL
        context_str = "\n\n".join(
            f"Document {i+1}:\n{doc}" for i, doc in enumerate(docs)
        )

        prompt = (
            "A list of documents is shown below. Each document has a number next to it along "
            "with a summary of the document. A question is also provided. \n"
            "Respond with the numbers of the documents you should consult to answer the question, "
            "in order of relevance, as well as the relevance score. The relevance score is a number "
            "from 1-10 based on how relevant you think the document is to the question.\n"
            f"Do not include any documents that are not relevant to the question or ranks less than {min_score}.\n"
            "Example format: \n"
            "Document 1:\n<summary of document 1>\n\n"
            "Document 2:\n<summary of document 2>\n\n"
            "...\n\n"
            "Document 10:\n<summary of document 10>\n\n"
            "Question: <question>\n"
            "Answer:\n"
            "Doc: 9, Relevance: 7\n"
            "Doc: 3, Relevance: 4\n"
            "Doc: 7, Relevance: 3\n\n"
            "Let's try this now:\n\n"
            f"{context_str}\n"
            f"Question: {query}\n"
            "Answer:\n"
        )

        messages: List[ChatCompletionSystemMessageParam | ChatCompletionUserMessageParam] = [
            ChatCompletionSystemMessageParam(role="system",
                                             content="You are a helpful assistant that ranks documents by relevance."),
            ChatCompletionUserMessageParam(role="user", content=prompt),
        ]

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )

        content = resp.choices[0].message.content or ""

        # Parsear líneas tipo: "Doc: 2, Relevance: 9"
        pattern = r"Doc:\s*(\d+),\s*Relevance:\s*(\d+)"
        items = [
            {"index": int(i), "score": int(score)}
            for i, score in re.findall(pattern, content)
        ]

        # Filtrar resultados válidos
        scored_docs = [
            (docs[item["index"] - 1], item["score"])
            for item in items
            if 1 <= item["index"] <= len(docs)
        ]

        # Ordenar por score descendente
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        return scored_docs
