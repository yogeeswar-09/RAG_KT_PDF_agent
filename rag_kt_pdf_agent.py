import os
import logging
import tempfile

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File
from langserve import add_routes
import uvicorn

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_pdf_agent")

GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY or GEMINI_API_KEY is missing."
    )


# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_retries=0,
    timeout=30,
)


# ============================================================
# EMBEDDINGS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)


# ============================================================
# GLOBAL VECTOR STORE
# ============================================================

vector_store = None

current_pdf_name = None


# ============================================================
# PDF PROCESSING
# ============================================================

def process_pdf(file_path: str):
    """
    Load PDF, split text into chunks and create
    an in-memory FAISS vector store.
    """

    global vector_store

    logger.info(
        "Loading PDF: %s",
        file_path
    )

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    if not documents:
        raise ValueError(
            "The PDF does not contain readable text."
        )

    logger.info(
        "PDF pages loaded: %s",
        len(documents)
    )

    # --------------------------------------------------------
    # Split PDF text
    # --------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    chunks = splitter.split_documents(
        documents
    )

    if not chunks:
        raise ValueError(
            "No text chunks could be created from the PDF."
        )

    logger.info(
        "Created %s chunks",
        len(chunks)
    )

    # --------------------------------------------------------
    # Create FAISS vector store
    # --------------------------------------------------------

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    logger.info(
        "FAISS vector store created."
    )

    return len(documents), len(chunks)


# ============================================================
# PDF UPLOAD ENDPOINT
# ============================================================

@app = None
