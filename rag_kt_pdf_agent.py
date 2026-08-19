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
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_kt_pdf_agent")

GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY or GEMINI_API_KEY is missing."
    )


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="RAG Agent - PDF Parsing",
    version="1.0.0",
    description=(
        "PDF parsing and Retrieval-Augmented Generation "
        "agent using LangChain, FAISS and Gemini."
    ),
)


# ============================================================
# GEMINI
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
# GLOBAL RAG STATE
# ============================================================

vector_store = None
current_pdf_name = None
current_page_count = 0
current_chunk_count = 0


# ============================================================
# PDF PROCESSING
# ============================================================

def process_pdf(file_path: str):

    global vector_store
    global current_page_count
    global current_chunk_count

    logger.info("Loading PDF...")

    loader = PyPDFLoader(file_path)

    documents = loader.load()

    if not documents:
        raise ValueError(
            "The PDF contains no readable text."
        )

    current_page_count = len(documents)

    logger.info(
        "Loaded %d pages.",
        current_page_count
    )

    # --------------------------------------------------------
    # TEXT CHUNKING
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
            "Could not create text chunks from PDF."
        )

    current_chunk_count = len(chunks)

    logger.info(
        "Created %d chunks.",
        current_chunk_count
    )

    # --------------------------------------------------------
    # VECTOR DATABASE
    # --------------------------------------------------------

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    logger.info(
        "FAISS vector store created."
    )

    return (
        current_page_count,
        current_chunk_count
    )


# ============================================================
# PDF UPLOAD
# ============================================================

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...)
):

    global current_pdf_name

    if not file.filename:
        return {
            "success": False,
            "error": "No file selected."
        }

    if not file.filename.lower().endswith(".pdf"):
        return {
            "success": False,
            "error": "Only PDF files are supported."
        }

    temp_path = None

    try:

        pdf_data = await file.read()

        if not pdf_data:
            return {
                "success": False,
                "error": "Uploaded PDF is empty."
            }

        # ----------------------------------------------------
        # Temporary PDF
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(pdf_data)
            temp_path = temp_file.name

        # ----------------------------------------------------
        # Parse and Index
        # ----------------------------------------------------

        pages, chunks = process_pdf(
            temp_path
        )

        current_pdf_name = file.filename

        return {
            "success": True,
            "message": "PDF processed successfully.",
            "filename": current_pdf_name,
            "pages": pages,
            "chunks": chunks,
            "ready": True,
            "playground": "/agent/playground/"
        }

    except Exception as error:

        logger.exception(
            "PDF processing failed."
        )

        return {
            "success": False,
            "error": str(error)
        }

    finally:

        if temp_path:

            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================
# RAG PROMPT
# ============================================================

rag_prompt = ChatPromptTemplate.from_template(
    """
You are a PDF question-answering assistant.

Your job is to answer the user's question using ONLY
the information contained in the supplied PDF context.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent information.
3. If the answer is not present in the context, say:
   "I couldn't find that information in the uploaded PDF."
4. Keep the answer clear and natural.
5. When possible, mention the relevant page number.
6. Do not reveal chain-of-thought.

PDF CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""
)


# ============================================================
# FORMAT RETRIEVED DOCUMENTS
# ============================================================

def format_documents(documents):

    if not documents:
        return (
            "No relevant content was found "
            "in the uploaded PDF."
        )

    output = []

    for document in documents:

        page = document.metadata.get(
            "page"
        )

        if page is not None:

            page_number = page + 1

            output.append(
                f"[Page {page_number}]\n"
                f"{document.page_content}"
            )

        else:

            output.append(
                document.page_content
            )

    return "\n\n".join(output)


# ============================================================
# RAG QUESTION ANSWERING
# ============================================================

def answer_question(
    question: str
) -> str:

    global vector_store
    global current_pdf_name

    question = str(
        question or ""
    ).strip()

    if not question:

        return (
            "Please enter a question about "
            "the uploaded PDF."
        )

    if vector_store is None:

        return (
            "Please upload a PDF first. "
            "The RAG knowledge base is empty."
        )

    logger.info(
        "Question: %s",
        question
    )

    try:

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        documents = vector_store.similarity_search(
            question,
            k=4
        )

        context = format_documents(
            documents
        )

        # ----------------------------------------------------
        # GENERATION
        # ----------------------------------------------------

        chain = (
            rag_prompt
            | llm
            | StrOutputParser()
        )

        answer = chain.invoke(
            {
                "context": context,
                "question": question
            }
        )

        source_name = (
            current_pdf_name
            or "Uploaded PDF"
        )

        return (
            f"### Answer\n\n"
            f"{answer}\n\n"
            f"---\n\n"
            f"**Source:** {source_name}\n"
            f"**Retrieved chunks:** "
            f"{len(documents)}"
        )

    except Exception as error:

        logger.exception(
            "RAG question failed."
        )

        return (
            "RAG Agent error: "
            f"{error}"
        )


# ============================================================
# LANGSERVE
# ============================================================

rag_runnable = RunnableLambda(
    answer_question
)

add_routes(
    app,
    rag_runnable,
    path="/agent",
    input_type=str,
    output_type=str,
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "agent": "RAG Agent",
        "type": "PDF Parsing + RAG",
        "framework": "LangChain",
        "status": "running",
        "pdf_ready": vector_store is not None,
        "current_pdf": current_pdf_name,
        "pages": current_page_count,
        "chunks": current_chunk_count,
        "upload_endpoint": "/upload-pdf",
        "playground": "/agent/playground/"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "agent": "RAG Agent",
        "pdf_ready": vector_store is not None
    }


# ============================================================
# SERVER
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                "8000"
            )
        )
    )
