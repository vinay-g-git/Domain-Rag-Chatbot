"""
rag_pipeline.py
----------------
Module 5 (Retrieval) glue + Module 6 (Answer Generation).

Ties document_loader.py -> vector_store.py -> an LLM provider together, and
enforces the "answer only from context" guardrail from prompt.py.

Supports multiple LLM providers (Groq, OpenAI, Gemini) chosen via the
LLM_PROVIDER environment variable, so students can use whichever free/paid
API key they have.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from document_loader import extract_pages_from_multiple_pdfs
from prompt import build_system_prompt, build_user_message
from vector_store import VectorStore, split_pages_into_chunks, Chunk, DEFAULT_TOP_K

NOT_FOUND_MESSAGE = "I could not find this information in the uploaded documents."


@dataclass
class RetrievedSource:
    document: str
    page: int
    score: float
    snippet: str


@dataclass
class RAGAnswer:
    answer: str
    sources: list[RetrievedSource]


class LLMError(RuntimeError):
    """Raised when the configured LLM provider fails to respond."""


def _call_groq(system_prompt: str, user_message: str) -> str:
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    resp = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


def _call_openai(system_prompt: str, user_message: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content


def _call_gemini(system_prompt: str, user_message: str) -> str:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel(
        os.environ.get("GEMINI_MODEL", "gemini-1.5-flash"),
        system_instruction=system_prompt,
    )
    resp = model.generate_content(user_message)
    return resp.text


PROVIDERS = {
    "groq": _call_groq,
    "openai": _call_openai,
    "gemini": _call_gemini,
}


def call_llm(system_prompt: str, user_message: str) -> str:
    """Dispatch to whichever provider is configured in LLM_PROVIDER."""
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    fn = PROVIDERS.get(provider)
    if fn is None:
        raise LLMError(
            f"Unknown LLM_PROVIDER '{provider}'. Choose one of: {', '.join(PROVIDERS)}"
        )
    try:
        return fn(system_prompt, user_message)
    except KeyError as e:
        raise LLMError(f"Missing API key for provider '{provider}': {e}") from e
    except Exception as e:
        raise LLMError(f"LLM call to '{provider}' failed: {e}") from e


class RAGChatbot:
    """End-to-end pipeline: build an index from PDFs, then answer questions."""

    def __init__(self, domain: str = "General", top_k: int = DEFAULT_TOP_K):
        self.domain = domain
        self.top_k = top_k
        self.store = VectorStore()

    def ingest(self, pdf_paths: list[str]) -> int:
        """Extract, chunk, and embed a set of PDFs. Returns chunk count."""
        pages = extract_pages_from_multiple_pdfs(pdf_paths)
        chunks: list[Chunk] = split_pages_into_chunks(pages)
        self.store.build(chunks)
        return len(chunks)

    def save_index(self, directory: str) -> None:
        self.store.save(directory)

    def load_index(self, directory: str) -> None:
        self.store = VectorStore.load(directory)

    def ask(self, question: str) -> RAGAnswer:
        """Retrieve relevant chunks and generate a grounded answer."""
        results = self.store.search(question, top_k=self.top_k)

        if not results:
            return RAGAnswer(answer=NOT_FOUND_MESSAGE, sources=[])

        context_blocks = [
            f"[{c.source_document}, page {c.page_number}]\n{c.text}" for c, _ in results
        ]
        system_prompt = build_system_prompt(self.domain)
        user_message = build_user_message(question, context_blocks)

        try:
            answer_text = call_llm(system_prompt, user_message)
        except LLMError as e:
            # Fail safe rather than pretending we generated an answer.
            return RAGAnswer(answer=f"(LLM error) {e}", sources=[])

        sources = [
            RetrievedSource(
                document=c.source_document,
                page=c.page_number,
                score=score,
                snippet=c.text[:200],
            )
            for c, score in results
        ]
        return RAGAnswer(answer=answer_text.strip(), sources=sources)
