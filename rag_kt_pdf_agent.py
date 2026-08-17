import os
import shutil
import tempfile
import logging

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from langserve import add_routes
import uvicorn

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# ----------------------------
# Load Environment
# ----------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_kt_agent")

# ----------------------------
# Initialize LLM & Embeddings
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

# ----------------------------
# Global Vector Store
# ----------------------------
vector_store = None

# ----------------------------
# PDF Processing
# ----------------------------
def load_and_split_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    if not pages:
        raise ValueError("The PDF did not contain readable text.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata["source"] = os.path.basename(pdf_path)

    return chunks


def build_vector_store(chunks):
    logger.info("Building FAISS vector store...")
    return FAISS.from_documents(chunks, embeddings)


# ----------------------------
# RAG Prompt
# ----------------------------
rag_prompt = ChatPromptTemplate.from_template(
    """
You are a Knowledge Transfer assistant.

Answer the user's question using ONLY the provided PDF context.

Rules:
- Do not invent information.
- If the answer is not present in the context, clearly say that
  the uploaded document does not contain the answer.
- Give a concise but useful answer.
- Mention relevant source pages when possible.

Context:
{context}

Question:
{question}

Answer:
"""
)


def format_docs(docs):
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, "
        f"Page: {doc.metadata.get('page', '?')}]\n"
        f"{doc.page_content}"
        for doc in docs
    )


def run_rag(request):
    global vector_store

    question = request.get("input", "")

    if not question:
        return "Please provide a question."

    if vector_store is None:
        return (
            "No PDF has been uploaded yet. "
            "Upload a PDF using POST /upload-pdf first."
        )

    try:
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 4}
        )

        docs = retriever.invoke(question)

        context = format_docs(docs)

        messages = rag_prompt.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        response = llm.invoke(messages)

        return StrOutputParser().invoke(response)

    except Exception as exc:
        logger.exception("RAG query failed")
        return f"RAG error: {exc}"


rag_runnable = RunnableLambda(run_rag)

# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(
    title="KT PDF RAG Agent",
    version="1.0",
    description=(
        "Knowledge Transfer RAG agent that parses uploaded PDFs, "
        "creates Gemini embeddings, stores them in FAISS, "
        "and answers questions using retrieved document context."
    ),
)

# ----------------------------
# LangServe RAG Route
# ----------------------------
add_routes(
    app,
    rag_runnable,
    path="/agent",
)

# ----------------------------
# PDF Upload Endpoint
# ----------------------------
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global vector_store

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:
            temp_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        chunks = load_and_split_pdf(temp_path)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No readable text was found in the PDF.",
            )

        vector_store = build_vector_store(chunks)

        return {
            "status": "success",
            "filename": file.filename,
            "pages_and_chunks_processed": len(chunks),
            "message": (
                "PDF processed successfully. "
                "You can now query POST /agent/invoke."
            ),
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("PDF processing failed")
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {exc}",
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ----------------------------
# Health Check
# ----------------------------
@app.get("/")
def root():
    return {
        "agent": "KT PDF RAG Agent",
        "status": "running",
        "upload_endpoint": "/upload-pdf",
        "query_endpoint": "/agent",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "pdf_loaded": vector_store is not None,
    }


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(
        "rag_kt_pdf_agent:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
