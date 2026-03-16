"""
RAG over Excel Sheets — OpenAI + LangChain
==========================================
Replaces:  Docling  →  pandas + openpyxl
           Llama-3.2 (local) →  OpenAI GPT-4o-mini
           LlamaIndex         →  LangChain + FAISS
"""

import os
import pandas as pd
from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY from .env
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory


# ─────────────────────────────────────────────
# Step 1 — Parse Excel file into LangChain Docs
# ─────────────────────────────────────────────

def load_excel_as_documents(file_path: str) -> List[Document]:
    """
    Reads every sheet in an Excel workbook.
    Each sheet is converted to a markdown-style text block and
    wrapped in a LangChain Document with sheet metadata.
    """
    xl = pd.ExcelFile(file_path)
    documents: List[Document] = []

    for sheet_name in xl.sheet_names:
        df = xl.parse(sheet_name)

        # Drop completely empty rows/columns
        df.dropna(how="all", inplace=True)
        df.dropna(axis=1, how="all", inplace=True)

        if df.empty:
            continue

        # Convert the sheet to a readable text representation
        # markdown table → keeps column headers, easy for LLM to parse
        table_text = df.to_markdown(index=False)

        # Also add a plain CSV fallback for numeric-heavy sheets
        csv_text = df.to_csv(index=False)

        content = (
            f"Sheet: {sheet_name}\n\n"
            f"### Table (Markdown)\n{table_text}\n\n"
            f"### Raw CSV\n{csv_text}"
        )

        doc = Document(
            page_content=content,
            metadata={"source": file_path, "sheet": sheet_name},
        )
        documents.append(doc)

    print(f"✅ Loaded {len(documents)} sheet(s) from '{file_path}'")
    return documents


# ─────────────────────────────────────────────
# Step 2 — Chunk documents
# ─────────────────────────────────────────────

def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits large sheet documents into overlapping chunks so that
    the retriever can surface the most relevant rows/sections.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", "| ", ",", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Split into {len(chunks)} chunk(s)")
    return chunks


# ─────────────────────────────────────────────
# Step 3 — Embed and build FAISS vector store
# ─────────────────────────────────────────────

def build_vector_store(chunks: List[Document], openai_api_key: str) -> FAISS:
    """
    Embeds chunks using OpenAI text-embedding-3-small and stores
    them in an in-memory FAISS index.
    """
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key,
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("✅ Vector store built (FAISS)")
    return vector_store


# ─────────────────────────────────────────────
# Step 4 — Build conversational RAG chain (LCEL)
# ─────────────────────────────────────────────

# Per-session in-memory store  { session_id: ChatMessageHistory }
_session_store: dict = {}

def _get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def build_rag_chain(vector_store: FAISS, openai_api_key: str) -> RunnableWithMessageHistory:
    """
    Wires together:
      • FAISS retriever  (top-4 chunks)
      • GPT-4o-mini LLM
      • Per-session ChatMessageHistory
    using the modern LCEL RunnableWithMessageHistory pattern.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=openai_api_key,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful data analyst. Use the retrieved Excel context below "
         "to answer the user's question accurately and concisely.\n\n"
         "Context:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def retrieve_context(inputs: dict) -> dict:
        docs = retriever.invoke(inputs["question"])
        inputs["context"] = "\n\n".join(d.page_content for d in docs)
        inputs["_source_docs"] = docs
        return inputs

    chain = (
        RunnablePassthrough()
        | retrieve_context
        | prompt
        | llm
        | StrOutputParser()
    )

    chain_with_history = RunnableWithMessageHistory(
        chain,                                        # type: ignore[arg-type]
        _get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    print("✅ RAG chain ready (LCEL)")
    return chain_with_history


# ─────────────────────────────────────────────
# Convenience: build everything in one call
# ─────────────────────────────────────────────

def build_pipeline(file_path: str, openai_api_key: str) -> RunnableWithMessageHistory:
    docs   = load_excel_as_documents(file_path)
    chunks = split_documents(docs)
    vs     = build_vector_store(chunks, openai_api_key)
    chain  = build_rag_chain(vs, openai_api_key)
    return chain


# ─────────────────────────────────────────────
# CLI quick-test  (python rag_pipeline.py)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    api_key   = os.environ.get("OPENAI_API_KEY", "")
    excel_path = sys.argv[1] if len(sys.argv) > 1 else "sample.xlsx"

    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Add it to your .env file.")

    chain = build_pipeline(excel_path, api_key)

    print("\n💬 Chat with your Excel data (type 'exit' to quit)\n")
    session_cfg = {"configurable": {"session_id": "cli-session"}}
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        answer = chain.invoke({"question": question}, config=session_cfg)
        print(f"\nAssistant: {answer}\n")