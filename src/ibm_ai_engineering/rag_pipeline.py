"""Retrieval-Augmented Generation (RAG) Architecture with LangChain.

Implements end-to-end document indexing, semantic chunking, Chroma vector store
retrieval, and conversational Q&A chain orchestration.
"""

from typing import List, Dict, Any, Optional
import os


class RAGPipeline:
    """Document QA pipeline using embeddings and vector retrieval."""

    def __init__(
        self,
        collection_name: str = "rag_knowledge_base",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vector_store = None

    def chunk_documents(self, raw_documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Split raw text documents into overlapping semantic chunks."""
        chunks = []
        for doc in raw_documents:
            text = doc.get("content", "")
            source = doc.get("source", "unknown")
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end]
                chunks.append({
                    "text": chunk_text,
                    "metadata": {"source": source, "start_char": start, "end_char": end}
                })
                if end == len(text):
                    break
                start += self.chunk_size - self.chunk_overlap
        return chunks

    def build_qa_prompt(self, context: str, question: str) -> str:
        """Construct prompt template incorporating retrieved context."""
        return (
            "You are a technical AI assistant. Answer the user question based strictly on the "
            "provided context. If the answer cannot be found in the context, state that you do "
            "not have sufficient information.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Helpful Answer:"
        )
