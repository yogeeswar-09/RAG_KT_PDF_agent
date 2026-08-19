import os
import logging
import tempfile
import html

from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse

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

logging.basicConfig(
    level=logging.INFO
)

logger = logging.getLogger(
    "rag_kt_pdf_agent"
)

GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY or GEMINI_API_KEY is missing."
    )


# ============================================================
# CONFIGURATION
# ============================================================

LLM_MODEL = os.getenv(
    "RAG_LLM_MODEL",
    "gemini-3.6-flash"
)

MAX_PDF_SIZE = 10 * 1024 * 1024


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="RAG PDF Agent",
    version="2.0.0",
    description=(
        "PDF parsing and Retrieval-Augmented Generation "
        "agent built with LangChain and Gemini."
    ),
)


# ============================================================
# GEMINI CHAT MODEL
# ============================================================

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=0,
    max_retries=0,
    timeout=60,
)


# ============================================================
# GEMINI EMBEDDINGS
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
# RAG PROMPT
# ============================================================

rag_prompt = ChatPromptTemplate.from_template(
    """
You are a Retrieval-Augmented Generation assistant.

You answer questions about an uploaded PDF.

STRICT RULES:

1. Use ONLY the supplied PDF context.
2. Do not use outside knowledge.
3. Do not invent facts.
4. If the answer is not available in the context,
   say exactly:

   "I couldn't find that information in the uploaded PDF."

5. Give a clear, human-friendly answer.
6. Use bullet points when useful.
7. If page information is available in the context,
   mention the relevant page numbers.
8. Do not expose chain-of-thought or hidden reasoning.

PDF CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""
)


# ============================================================
# CUSTOM WEB UI
# ============================================================

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>RAG PDF Agent</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    min-height: 100vh;

    font-family:
        Inter,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;

    background:
        radial-gradient(
            circle at top left,
            #18264f 0%,
            #080c18 42%,
            #050711 100%
        );

    color: #f8fafc;
}

.container {
    width: min(1100px, 92%);
    margin: 0 auto;
    padding: 55px 0 80px;
}

.badge {
    display: inline-flex;

    padding: 9px 16px;

    border: 1px solid rgba(120, 160, 255, 0.35);

    border-radius: 999px;

    background:
        rgba(70, 100, 190, 0.12);

    color: #9ec5ff;

    font-size: 14px;

    margin-bottom: 22px;
}

h1 {
    margin: 0;

    font-size: clamp(
        42px,
        7vw,
        76px
    );

    line-height: 1.02;

    letter-spacing: -3px;
}

.gradient {
    background:
        linear-gradient(
            90deg,
            #60a5fa,
            #818cf8,
            #c084fc
        );

    -webkit-background-clip: text;
    background-clip: text;

    color: transparent;
}

.subtitle {
    max-width: 760px;

    margin-top: 22px;

    color: #9fb0ce;

    font-size: 18px;

    line-height: 1.7;
}

.card {
    margin-top: 40px;

    padding: 28px;

    border-radius: 24px;

    border: 1px solid
        rgba(148, 163, 184, 0.16);

    background:
        rgba(15, 23, 42, 0.72);

    backdrop-filter: blur(18px);

    box-shadow:
        0 30px 80px
        rgba(0, 0, 0, 0.35);
}

.section-title {
    font-size: 18px;

    font-weight: 700;

    margin-bottom: 14px;
}

.dropzone {
    min-height: 190px;

    border: 1.5px dashed
        rgba(96, 165, 250, 0.45);

    border-radius: 18px;

    display: flex;

    align-items: center;

    justify-content: center;

    text-align: center;

    padding: 25px;

    cursor: pointer;

    background:
        rgba(15, 23, 42, 0.6);

    transition:
        0.2s ease;
}

.dropzone:hover {
    border-color: #60a5fa;

    background:
        rgba(37, 99, 235, 0.08);
}

.dropzone.dragging {
    border-color: #a78bfa;

    background:
        rgba(124, 58, 237, 0.12);
}

.upload-icon {
    font-size: 45px;

    margin-bottom: 10px;
}

.drop-title {
    font-size: 19px;

    font-weight: 700;
}

.drop-subtitle {
    margin-top: 7px;

    color: #8493ae;

    font-size: 14px;
}

.filename {
    margin-top: 15px;

    color: #7dd3fc;

    font-weight: 600;

    word-break: break-word;
}

input[type="file"] {
    display: none;
}

.button-row {
    display: flex;

    gap: 12px;

    margin-top: 18px;
}

button {
    border: 0;

    border-radius: 13px;

    padding: 14px 22px;

    font-size: 15px;

    font-weight: 700;

    cursor: pointer;

    transition:
        transform 0.15s ease,
        opacity 0.15s ease;
}

button:hover {
    transform: translateY(-1px);
}

button:disabled {
    cursor: not-allowed;

    opacity: 0.45;

    transform: none;
}

.primary {
    flex: 1;

    color: white;

    background:
        linear-gradient(
            90deg,
            #2563eb,
            #4f46e5,
            #7c3aed
        );
}

.secondary {
    color: #cbd5e1;

    background:
        rgba(30, 41, 59, 0.9);

    border: 1px solid
        rgba(148, 163, 184, 0.18);
}

.status {
    display: none;

    margin-top: 18px;

    padding: 15px 17px;

    border-radius: 13px;

    font-size: 14px;

    line-height: 1.5;
}

.status.success {
    display: block;

    color: #a7f3d0;

    background:
        rgba(16, 185, 129, 0.09);

    border:
        1px solid
        rgba(16, 185, 129, 0.2);
}

.status.error {
    display: block;

    color: #fecaca;

    background:
        rgba(239, 68, 68, 0.09);

    border:
        1px solid
        rgba(239, 68, 68, 0.2);
}

.status.loading {
    display: block;

    color: #bfdbfe;

    background:
        rgba(59, 130, 246, 0.08);

    border:
        1px solid
        rgba(59, 130, 246, 0.18);
}

.stats {
    display: none;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 12px;

    margin-top: 20px;
}

.stat {
    padding: 18px;

    border-radius: 15px;

    background:
        rgba(15, 23, 42, 0.8);

    border:
        1px solid
        rgba(148, 163, 184, 0.12);
}

.stat-number {
    font-size: 25px;

    font-weight: 800;
}

.stat-label {
    margin-top: 5px;

    color: #7f8da7;

    font-size: 13px;
}

.question-section {
    display: none;

    margin-top: 35px;
}

textarea {
    width: 100%;

    min-height: 115px;

    resize: vertical;

    border-radius: 15px;

    border: 1px solid
        rgba(148, 163, 184, 0.18);

    background:
        rgba(2, 6, 23, 0.8);

    color: #f8fafc;

    padding: 17px;

    font-family: inherit;

    font-size: 16px;

    outline: none;
}

textarea:focus {
    border-color: #60a5fa;

    box-shadow:
        0 0 0 3px
        rgba(96, 165, 250, 0.1);
}

.answer {
    display: none;

    margin-top: 25px;

    padding: 25px;

    border-radius: 18px;

    border: 1px solid
        rgba(129, 140, 248, 0.2);

    background:
        rgba(15, 23, 42, 0.85);

    line-height: 1.75;

    color: #dbeafe;
}

.answer h3 {
    margin-top: 0;

    color: white;
}

.source {
    margin-top: 20px;

    padding-top: 15px;

    border-top:
        1px solid
        rgba(148, 163, 184, 0.12);

    color: #8191ac;

    font-size: 13px;
}

.links {
    margin-top: 28px;

    display: flex;

    gap: 22px;

    flex-wrap: wrap;
}

.links a {
    color: #60a5fa;

    text-decoration: none;

    font-size: 14px;
}

.links a:hover {
    text-decoration: underline;
}

.footer {
    text-align: center;

    margin-top: 45px;

    color: #526078;

    font-size: 13px;
}

@media(max-width: 700px) {

    .container {
        padding-top: 30px;
    }

    .card {
        padding: 20px;
    }

    .stats {
        grid-template-columns: 1fr;
    }

    .button-row {
        flex-direction: column;
    }

    h1 {
        letter-spacing: -2px;
    }

}

</style>

</head>


<body>

<div class="container">

    <div class="badge">
        ✦ LangChain · Gemini · FAISS · PDF RAG
    </div>

    <h1>
        RAG PDF
        <span class="gradient">Agent</span>
    </h1>

    <p class="subtitle">
        Upload a PDF, let the agent understand its contents,
        and ask questions using retrieval-augmented generation.
        Answers are grounded in the uploaded document.
    </p>


    <div class="card">

        <!-- ============================================ -->
        <!-- UPLOAD -->
        <!-- ============================================ -->

        <div class="section-title">
            📄 Upload your PDF
        </div>

        <div
            class="dropzone"
            id="dropzone"
        >

            <div>

                <div class="upload-icon">
                    📄
                </div>

                <div class="drop-title">
                    Drop your PDF here
                </div>

                <div class="drop-subtitle">
                    or click to browse · PDF up to 10 MB
                </div>

                <div
                    class="filename"
                    id="filename"
                ></div>

            </div>

        </div>

        <input
            type="file"
            id="fileInput"
            accept=".pdf,application/pdf"
        >


        <div class="button-row">

            <button
                class="primary"
                id="uploadButton"
                disabled
            >
                Process PDF
            </button>

            <button
                class="secondary"
                id="clearButton"
            >
                Clear
            </button>

        </div>


        <div
            class="status"
            id="status"
        ></div>


        <!-- ============================================ -->
        <!-- STATS -->
        <!-- ============================================ -->

        <div
            class="stats"
            id="stats"
        >

            <div class="stat">

                <div
                    class="stat-number"
                    id="pageCount"
                >
                    0
                </div>

                <div class="stat-label">
                    PDF Pages
                </div>

            </div>


            <div class="stat">

                <div
                    class="stat-number"
                    id="chunkCount"
                >
                    0
                </div>

                <div class="stat-label">
                    Text Chunks
                </div>

            </div>


            <div class="stat">

                <div
                    class="stat-number"
                    id="readyStatus"
                >
                    —
                </div>

                <div class="stat-label">
                    RAG Status
                </div>

            </div>

        </div>


        <!-- ============================================ -->
        <!-- QUESTIONS -->
        <!-- ============================================ -->

        <div
            class="question-section"
            id="questionSection"
        >

            <div class="section-title">
                💬 Ask your PDF
            </div>

            <textarea
                id="question"
                placeholder="Example: What is the main topic of this document?"
            ></textarea>


            <div class="button-row">

                <button
                    class="primary"
                    id="askButton"
                >
                    Ask Question →
                </button>

            </div>


            <div
                class="answer"
                id="answer"
            ></div>

        </div>


        <!-- ============================================ -->
        <!-- LINKS -->
        <!-- ============================================ -->

        <div class="links">

            <a
                href="/agent/playground/"
                target="_blank"
            >
                LangServe Playground ↗
            </a>

            <a
                href="/docs"
                target="_blank"
            >
                API Docs ↗
            </a>

            <a
                href="/health"
                target="_blank"
            >
                Health ↗
            </a>

        </div>

    </div>


    <div class="footer">
        RAG PDF Agent · LangChain · Gemini · FAISS
    </div>

</div>


<script>

const dropzone =
    document.getElementById("dropzone");

const fileInput =
    document.getElementById("fileInput");

const uploadButton =
    document.getElementById("uploadButton");

const clearButton =
    document.getElementById("clearButton");

const filename =
    document.getElementById("filename");

const statusBox =
    document.getElementById("status");

const stats =
    document.getElementById("stats");

const pageCount =
    document.getElementById("pageCount");

const chunkCount =
    document.getElementById("chunkCount");

const readyStatus =
    document.getElementById("readyStatus");

const questionSection =
    document.getElementById("questionSection");

const question =
    document.getElementById("question");

const askButton =
    document.getElementById("askButton");

const answer =
    document.getElementById("answer");


let selectedFile = null;


// ========================================================
// FILE SELECTION
// ========================================================

dropzone.addEventListener(
    "click",
    () => fileInput.click()
);


fileInput.addEventListener(
    "change",
    () => {

        if (fileInput.files.length > 0) {

            selectFile(
                fileInput.files[0]
            );

        }

    }
);


// ========================================================
// DRAG AND DROP
// ========================================================

dropzone.addEventListener(
    "dragover",
    (event) => {

        event.preventDefault();

        dropzone.classList.add(
            "dragging"
        );

    }
);


dropzone.addEventListener(
    "dragleave",
    () => {

        dropzone.classList.remove(
            "dragging"
        );

    }
);


dropzone.addEventListener(
    "drop",
    (event) => {

        event.preventDefault();

        dropzone.classList.remove(
            "dragging"
        );

        const files =
            event.dataTransfer.files;

        if (files.length > 0) {

            selectFile(
                files[0]
            );

        }

    }
);


// ========================================================
// SELECT FILE
// ========================================================

function selectFile(file) {

    if (
        file.type !== "application/pdf"
        &&
        !file.name
            .toLowerCase()
            .endsWith(".pdf")
    ) {

        showStatus(
            "Please select a PDF file.",
            "error"
        );

        return;
    }


    if (
        file.size >
        10 * 1024 * 1024
    ) {

        showStatus(
            "PDF must be smaller than 10 MB.",
            "error"
        );

        return;
    }


    selectedFile = file;

    filename.textContent =
        "✓ " + file.name;

    uploadButton.disabled = false;

    showStatus(
        "PDF selected. Click Process PDF to index it.",
        "success"
    );

}


// ========================================================
// UPLOAD PDF
// ========================================================

uploadButton.addEventListener(
    "click",
    async () => {

        if (!selectedFile) {

            showStatus(
                "Please select a PDF first.",
                "error"
            );

            return;
        }


        uploadButton.disabled = true;

        showStatus(
            "⏳ Parsing PDF, creating chunks and building the vector index...",
            "loading"
        );


        const formData =
            new FormData();

        formData.append(
            "file",
            selectedFile
        );


        try {

            const response =
                await fetch(
                    "/upload-pdf",
                    {
                        method: "POST",
                        body: formData
                    }
                );


            const data =
                await response.json();


            if (
                !response.ok
                ||
                !data.success
            ) {

                throw new Error(
                    data.error
                    ||
                    "PDF processing failed."
                );

            }


            pageCount.textContent =
                data.pages;

            chunkCount.textContent =
                data.chunks;

            readyStatus.textContent =
                "READY";

            stats.style.display =
                "grid";

            questionSection.style.display =
                "block";


            showStatus(
                "✓ PDF processed successfully. Your RAG agent is ready.",
                "success"
            );


            question.focus();


        } catch (error) {

            showStatus(
                "Unable to process the PDF: "
                + error.message,
                "error"
            );

        } finally {

            uploadButton.disabled = false;

        }

    }
);


// ========================================================
// ASK QUESTION
// ========================================================

askButton.addEventListener(
    "click",
    askQuestion
);


question.addEventListener(
    "keydown",
    (event) => {

        if (
            event.key === "Enter"
            &&
            !event.shiftKey
        ) {

            event.preventDefault();

            askQuestion();

        }

    }
);


async function askQuestion() {

    const text =
        question.value.trim();


    if (!text) {

        showStatus(
            "Please enter a question.",
            "error"
        );

        return;
    }


    askButton.disabled = true;

    answer.style.display =
        "block";

    answer.innerHTML =
        "<h3>Thinking...</h3>"
        +
        "<p>Searching the PDF and generating an answer...</p>";


    try {

        const response =
            await fetch(
                "/ask",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body: JSON.stringify({
                        question: text
                    })
                }
            );


        const data =
            await response.json();


        if (
            !response.ok
            ||
            !data.success
        ) {

            throw new Error(
                data.error
                ||
                "Question failed."
            );

        }


        answer.innerHTML =
            data.html;


    } catch (error) {

        answer.innerHTML =
            "<h3>Something went wrong</h3>"
            +
            "<p>"
            +
            escapeHtml(
                error.message
            )
            +
            "</p>";

    } finally {

        askButton.disabled = false;

    }

}


// ========================================================
// CLEAR
// ========================================================

clearButton.addEventListener(
    "click",
    async () => {

        try {

            await fetch(
                "/clear",
                {
                    method: "POST"
                }
            );

        } catch (error) {

            console.log(error);

        }


        selectedFile = null;

        fileInput.value = "";

        filename.textContent = "";

        uploadButton.disabled = true;

        stats.style.display =
            "none";

        questionSection.style.display =
            "none";

        answer.style.display =
            "none";

        question.value = "";

        showStatus(
            "Knowledge base cleared.",
            "success"
        );

    }
);


// ========================================================
// STATUS
// ========================================================

function showStatus(
    message,
    type
) {

    statusBox.textContent =
        message;

    statusBox.className =
        "status " + type;

}


// ========================================================
// HTML ESCAPE
// ========================================================

function escapeHtml(
    text
) {

    return text
        .replace(
            /&/g,
            "&amp;"
        )
        .replace(
            /</g,
            "&lt;"
        )
        .replace(
            />/g,
            "&gt;"
        )
        .replace(
            /"/g,
            "&quot;"
        )
        .replace(
            /'/g,
            "&#039;"
        );

}

</script>

</body>

</html>
"""


# ============================================================
# ROOT UI
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home():

    return HTML_PAGE


# ============================================================
# PDF PROCESSING
# ============================================================

def process_pdf(
    file_path: str
):

    global vector_store
    global current_page_count
    global current_chunk_count

    logger.info(
        "Loading PDF: %s",
        file_path
    )

    loader = PyPDFLoader(
        file_path
    )

    documents = loader.load()

    if not documents:

        raise ValueError(
            "The PDF contains no readable text."
        )


    current_page_count =
        len(documents)


    # ========================================================
    # TEXT CHUNKING
    # ========================================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )


    chunks = splitter.split_documents(
        documents
    )


    if not chunks:

        raise ValueError(
            "No readable text was found in the PDF."
        )


    current_chunk_count =
        len(chunks)


    logger.info(
        "Creating embeddings for %d chunks...",
        len(chunks)
    )


    # ========================================================
    # FAISS VECTOR STORE
    # ========================================================

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )


    logger.info(
        "PDF indexed successfully."
    )


    return (
        current_page_count,
        current_chunk_count
    )


# ============================================================
# UPLOAD ENDPOINT
# ============================================================

@app.post(
    "/upload-pdf"
)
async def upload_pdf(
    file: UploadFile = File(...)
):

    global current_pdf_name


    if not file.filename:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "No PDF was selected."
            }
        )


    if not file.filename.lower().endswith(
        ".pdf"
    ):

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Only PDF files are supported."
            }
        )


    temp_path = None


    try:

        pdf_data =
            await file.read()


        if not pdf_data:

            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "The uploaded PDF is empty."
                }
            )


        if len(pdf_data) > MAX_PDF_SIZE:

            return JSONResponse(
                status_code=413,
                content={
                    "success": False,
                    "error":
                        "PDF is too large. Maximum size is 10 MB."
                }
            )


        # ----------------------------------------------------
        # Save temporary file
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(
                pdf_data
            )

            temp_path =
                temp_file.name


        # ----------------------------------------------------
        # Parse + index
        # ----------------------------------------------------

        pages, chunks =
            process_pdf(
                temp_path
            )


        current_pdf_name =
            file.filename


        return {
            "success": True,
            "message":
                "PDF processed successfully.",
            "filename":
                current_pdf_name,
            "pages":
                pages,
            "chunks":
                chunks,
            "ready":
                True
        }


    except Exception as error:

        logger.exception(
            "PDF processing error."
        )


        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error":
                    "Could not process the PDF. "
                    "Make sure it contains selectable text."
            }
        )


    finally:

        if temp_path:

            try:

                os.remove(
                    temp_path
                )

            except Exception:

                pass


# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post(
    "/ask"
)
async def ask(
    payload: dict
):

    global vector_store
    global current_pdf_name


    question =
        str(
            payload.get(
                "question",
                ""
            )
        ).strip()


    if not question:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error":
                    "Please enter a question."
            }
        )


    if vector_store is None:

        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error":
                    "Please upload and process a PDF first."
            }
        )


    try:

        # ====================================================
        # RETRIEVAL
        # ====================================================

        documents =
            vector_store.similarity_search(
                question,
                k=4
            )


        if not documents:

            return {
                "success": True,
                "answer":
                    "I couldn't find that information "
                    "in the uploaded PDF.",
                "html":
                    """
                    <h3>Answer</h3>
                    <p>
                    I couldn't find that information
                    in the uploaded PDF.
                    </p>
                    """
            }


        # ====================================================
        # BUILD CONTEXT
        # ====================================================

        context_parts = []

        pages = []


        for document in documents:

            page =
                document.metadata.get(
                    "page"
                )


            if page is not None:

                page_number =
                    page + 1

                pages.append(
                    page_number
                )


                context_parts.append(
                    f"[Page {page_number}]\n"
                    f"{document.page_content}"
                )

            else:

                context_parts.append(
                    document.page_content
                )


        context =
            "\n\n".join(
                context_parts
            )


        # ====================================================
        # GENERATION
        # ====================================================

        chain = (
            rag_prompt
            | llm
            | StrOutputParser()
        )


        result =
            chain.invoke(
                {
                    "context":
                        context,
                    "question":
                        question
                }
            )


        result =
            str(result).strip()


        # ====================================================
        # SOURCE PAGES
        # ====================================================

        unique_pages =
            sorted(
                set(pages)
            )


        if unique_pages:

            source_text =
                ", ".join(
                    str(page)
                    for page in unique_pages
                )

            source_html =
                (
                    "<div class='source'>"
                    "<strong>Source pages:</strong> "
                    + source_text
                    + "</div>"
                )

        else:

            source_html = ""


        safe_result =
            escape_html(
                result
            )


        # Convert simple markdown-like formatting
        safe_result =
            safe_result.replace(
                "\n",
                "<br>"
            )


        answer_html = (
            "<h3>Answer</h3>"
            "<p>"
            + safe_result
            + "</p>"
            + source_html
        )


        return {
            "success":
                True,

            "answer":
                result,

            "pages":
                unique_pages,

            "html":
                answer_html
        }


    except Exception as error:

        logger.exception(
            "Question answering failed."
        )


        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error":
                    "The RAG model could not answer "
                    "the question right now. "
                    "Please try again."
            }
        )


# ============================================================
# CLEAR VECTOR STORE
# ============================================================

@app.post(
    "/clear"
)
async def clear():

    global vector_store
    global current_pdf_name
    global current_page_count
    global current_chunk_count


    vector_store = None

    current_pdf_name = None

    current_page_count = 0

    current_chunk_count = 0


    return {
        "success":
            True,

        "message":
            "RAG knowledge base cleared."
    }


# ============================================================
# HEALTH
# ============================================================

@app.get(
    "/health"
)
async def health():

    return {
        "status":
            "healthy",

        "agent":
            "RAG PDF Agent",

        "pdf_ready":
            vector_store is not None,

        "pdf":
            current_pdf_name,

        "pages":
            current_page_count,

        "chunks":
            current_chunk_count,

        "model":
            LLM_MODEL
    }


# ============================================================
# API INFORMATION
# ============================================================

@app.get(
    "/api-info"
)
async def api_info():

    return {
        "agent":
            "RAG PDF Agent",

        "upload":
            "POST /upload-pdf",

        "ask":
            "POST /ask",

        "clear":
            "POST /clear",

        "health":
            "GET /health",

        "playground":
            "/agent/playground/",

        "docs":
            "/docs"
    }


# ============================================================
# LANGSERVE PLAYGROUND
# ============================================================

def playground_rag(
    question: str
):

    global vector_store


    if vector_store is None:

        return (
            "Please upload a PDF through the main "
            "RAG Agent page first:\n\n"
            "/"
        )


    question =
        str(
            question or ""
        ).strip()


    if not question:

        return (
            "Please enter a question."
        )


    try:

        documents =
            vector_store.similarity_search(
                question,
                k=4
            )


        if not documents:

            return (
                "I couldn't find that information "
                "in the uploaded PDF."
            )


        context_parts = []


        for document in documents:

            page =
                document.metadata.get(
                    "page"
                )


            if page is not None:

                context_parts.append(
                    f"[Page {page + 1}]\n"
                    f"{document.page_content}"
                )

            else:

                context_parts.append(
                    document.page_content
                )


        context =
            "\n\n".join(
                context_parts
            )


        chain = (
            rag_prompt
            | llm
            | StrOutputParser()
        )


        result =
            chain.invoke(
                {
                    "context":
                        context,

                    "question":
                        question
                }
            )


        return str(result)


    except Exception:

        return (
            "The RAG agent could not process "
            "the question right now."
        )


rag_runnable =
    RunnableLambda(
        playground_rag
    )


add_routes(
    app,
    rag_runnable,
    path="/agent",
    input_type=str,
    output_type=str
)


# ============================================================
# START SERVER
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
