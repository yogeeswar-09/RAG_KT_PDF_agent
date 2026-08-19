import os
import html
import logging
import tempfile

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from langserve import add_routes
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_pdf_agent")

API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY or GEMINI_API_KEY environment variable")

MODEL = os.getenv("RAG_LLM_MODEL", "gemini-3.6-flash")
MAX_FILE_SIZE = 10 * 1024 * 1024

app = FastAPI(title="RAG KT PDF Agent", version="1.0.0")

llm = ChatGoogleGenerativeAI(
    model=MODEL,
    google_api_key=API_KEY,
    temperature=0,
    max_retries=0,
    timeout=60,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=API_KEY,
)

vector_store = None
pdf_name = None
page_count = 0
chunk_count = 0

PROMPT = ChatPromptTemplate.from_template(
    """You are a PDF question-answering assistant.

Answer ONLY from the supplied PDF context.
Do not use outside knowledge.
Do not invent facts.
If the answer is not in the context, say:
\"I couldn't find that information in the uploaded PDF.\"
Give a concise, natural answer and mention page numbers when available.

PDF CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""
)

chain = PROMPT | llm | StrOutputParser()

HTML = """
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAG PDF Agent</title>
<style>
*{box-sizing:border-box}body{margin:0;min-height:100vh;background:radial-gradient(circle at 10% 0%,#172554,#080b16 42%,#05060d);color:#f8fafc;font-family:Inter,system-ui,sans-serif}.wrap{width:min(1050px,92%);margin:auto;padding:50px 0 70px}.badge{display:inline-block;padding:8px 15px;border:1px solid #334b83;border-radius:999px;color:#9cc5ff;background:#101b38;font-size:14px}h1{font-size:clamp(42px,7vw,72px);line-height:1;letter-spacing:-3px;margin:20px 0 15px}.grad{background:linear-gradient(90deg,#60a5fa,#818cf8,#c084fc);-webkit-background-clip:text;background-clip:text;color:transparent}.sub{max-width:760px;color:#94a3b8;font-size:18px;line-height:1.7}.card{margin-top:35px;padding:28px;border:1px solid #24314d;border-radius:24px;background:rgba(15,23,42,.78);box-shadow:0 30px 80px #0005}.title{font-size:18px;font-weight:800;margin-bottom:13px}.drop{min-height:180px;border:1.5px dashed #46618f;border-radius:18px;display:flex;align-items:center;justify-content:center;text-align:center;padding:20px;cursor:pointer;background:#090e1b}.drop.drag{border-color:#60a5fa;background:#0c1830}.icon{font-size:42px}.drop strong{display:block;font-size:19px;margin-top:8px}.muted{color:#7f8da6;font-size:14px;margin-top:7px}.file{color:#7dd3fc;font-weight:700;margin-top:13px;word-break:break-word}input[type=file]{display:none}.row{display:flex;gap:12px;margin-top:16px}button{border:0;border-radius:13px;padding:14px 22px;font-weight:800;font-size:15px;cursor:pointer}button:disabled{opacity:.45;cursor:not-allowed}.primary{flex:1;color:#fff;background:linear-gradient(90deg,#2563eb,#4f46e5,#7c3aed)}.secondary{background:#111827;color:#cbd5e1;border:1px solid #263248}.status{display:none;margin-top:16px;padding:14px 16px;border-radius:13px;font-size:14px}.ok{display:block;background:#06261e;border:1px solid #145d4b;color:#a7f3d0}.err{display:block;background:#2b1116;border:1px solid #6b202c;color:#fecaca}.load{display:block;background:#0c1c36;border:1px solid #214b80;color:#bfdbfe}.stats{display:none;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:18px}.stat{padding:17px;border:1px solid #1f2c43;border-radius:15px;background:#0a1020}.num{font-size:25px;font-weight:900}.lab{font-size:13px;color:#71809a;margin-top:4px}.ask{display:none;margin-top:30px}textarea{width:100%;min-height:115px;resize:vertical;padding:16px;border-radius:15px;border:1px solid #26344e;background:#070b14;color:#f8fafc;font:inherit;font-size:16px;outline:none}.answer{display:none;margin-top:20px;padding:22px;border:1px solid #303d68;border-radius:17px;background:#0a1020;line-height:1.75;color:#dbeafe}.answer h3{margin-top:0;color:#fff}.source{margin-top:18px;padding-top:13px;border-top:1px solid #1d293d;color:#7f8da6;font-size:13px}.links{display:flex;gap:22px;flex-wrap:wrap;margin-top:25px}.links a{color:#60a5fa;text-decoration:none;font-size:14px}.footer{text-align:center;color:#536078;margin-top:38px;font-size:13px}@media(max-width:650px){.wrap{padding-top:30px}.card{padding:20px}.row{flex-direction:column}.stats{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<div class="badge">✦ LangChain · Gemini · FAISS · PDF RAG</div><h1>RAG PDF <span class="grad">Agent</span></h1>
<div class="sub">Upload a PDF, build a searchable knowledge base, and ask questions grounded in the document.</div>
<div class="card"><div class="title">📄 Upload your PDF</div>
<div class="drop" id="drop"><div><div class="icon">📄</div><strong>Drop your PDF here</strong><div class="muted">or click to browse · PDF up to 10 MB</div><div class="file" id="name"></div></div></div>
<input id="file" type="file" accept=".pdf,application/pdf"><div class="row"><button class="primary" id="process" disabled>Process PDF</button><button class="secondary" id="clear">Clear</button></div>
<div id="status" class="status"></div><div class="stats" id="stats"><div class="stat"><div class="num" id="pages">0</div><div class="lab">PDF Pages</div></div><div class="stat"><div class="num" id="chunks">0</div><div class="lab">Text Chunks</div></div><div class="stat"><div class="num" id="ready">—</div><div class="lab">RAG Status</div></div></div>
<div class="ask" id="ask"><div class="title">💬 Ask your PDF</div><textarea id="question" placeholder="Example: What is the main topic of this document?"></textarea><div class="row"><button class="primary" id="askBtn">Ask Question →</button></div><div class="answer" id="answer"></div></div>
<div class="links"><a href="/agent/playground/" target="_blank">LangServe Playground ↗</a><a href="/docs" target="_blank">API Docs ↗</a><a href="/health" target="_blank">Health ↗</a></div></div><div class="footer">RAG PDF Agent · PDF Parsing · Retrieval · Gemini</div></div>
<script>
const $=id=>document.getElementById(id);const drop=$("drop"),file=$("file"),processBtn=$("process"),clearBtn=$("clear"),nameBox=$("name"),status=$("status"),stats=$("stats"),pages=$("pages"),chunks=$("chunks"),ready=$("ready"),ask=$("ask"),question=$("question"),askBtn=$("askBtn"),answer=$("answer");let selected=null;
function show(m,c){status.textContent=m;status.className="status "+c}function esc(s){return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;").replaceAll("'","&#039;")}
function selectFile(f){if(!f)return;if(f.type!=="application/pdf"&&!f.name.toLowerCase().endsWith(".pdf")){show("Please select a PDF file.","err");return}if(f.size>10485760){show("PDF must be smaller than 10 MB.","err");return}selected=f;nameBox.textContent="✓ "+f.name;processBtn.disabled=false;show("PDF selected. Click Process PDF.","ok")}
drop.onclick=()=>file.click();file.onchange=()=>selectFile(file.files[0]);drop.ondragover=e=>{e.preventDefault();drop.classList.add("drag")};drop.ondragleave=()=>drop.classList.remove("drag");drop.ondrop=e=>{e.preventDefault();drop.classList.remove("drag");selectFile(e.dataTransfer.files[0])};
processBtn.onclick=async()=>{if(!selected)return;processBtn.disabled=true;show("⏳ Parsing PDF, chunking text and creating embeddings...","load");const fd=new FormData();fd.append("file",selected);try{const r=await fetch("/upload-pdf",{method:"POST",body:fd});const d=await r.json();if(!r.ok||!d.success)throw Error(d.error||"PDF processing failed.");pages.textContent=d.pages;chunks.textContent=d.chunks;ready.textContent="READY";stats.style.display="grid";ask.style.display="block";show("✓ PDF processed successfully. RAG is ready.","ok");question.focus()}catch(e){show("Unable to process PDF: "+e.message,"err")}finally{processBtn.disabled=false}};
askBtn.onclick=async()=>{const q=question.value.trim();if(!q){show("Please enter a question.","err");return}askBtn.disabled=true;answer.style.display="block";answer.innerHTML="<h3>Thinking...</h3><p>Searching the PDF and generating an answer...</p>";try{const r=await fetch("/ask",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q})});const d=await r.json();if(!r.ok||!d.success)throw Error(d.error||"Question failed.");answer.innerHTML=d.html}catch(e){answer.innerHTML="<h3>Something went wrong</h3><p>"+esc(e.message)+"</p>"}finally{askBtn.disabled=false}};
question.addEventListener("keydown",e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();askBtn.click()}});clearBtn.onclick=async()=>{try{await fetch("/clear",{method:"POST"})}catch(e){}selected=null;file.value="";nameBox.textContent="";processBtn.disabled=true;stats.style.display="none";ask.style.display="none";answer.style.display="none";question.value="";show("Knowledge base cleared.","ok")};
</script></body></html>
"""

@app.get("/", response_class=HTMLResponse)
async def home():
    return HTML


def build_index(pdf_path):
    global vector_store, page_count, chunk_count
    docs = PyPDFLoader(pdf_path).load()
    if not docs:
        raise ValueError("No readable text was found in the PDF.")
    page_count = len(docs)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError("No text chunks could be created from the PDF.")
    chunk_count = len(chunks)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return page_count, chunk_count


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    global pdf_name
    if not file.filename:
        return JSONResponse(status_code=400, content={"success": False, "error": "No file selected."})
    if not file.filename.lower().endswith(".pdf"):
        return JSONResponse(status_code=400, content={"success": False, "error": "Only PDF files are supported."})
    data = await file.read()
    if not data:
        return JSONResponse(status_code=400, content={"success": False, "error": "The PDF is empty."})
    if len(data) > MAX_FILE_SIZE:
        return JSONResponse(status_code=413, content={"success": False, "error": "PDF exceeds the 10 MB limit."})
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(data)
            temp_path = temp.name
        pages, chunks = build_index(temp_path)
        pdf_name = file.filename
        return {"success": True, "filename": pdf_name, "pages": pages, "chunks": chunks, "ready": True}
    except Exception as exc:
        logger.exception("PDF processing failed")
        return JSONResponse(status_code=500, content={"success": False, "error": f"PDF processing failed: {type(exc).__name__}: {exc}"})
    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


def retrieve(question, k=4):
    if vector_store is None:
        raise RuntimeError("Upload and process a PDF first.")
    return vector_store.similarity_search(question, k=k)


@app.post("/ask")
async def ask_question(payload: dict):
    question = str(payload.get("question", "")).strip()
    if not question:
        return JSONResponse(status_code=400, content={"success": False, "error": "Question is required."})
    try:
        docs = retrieve(question)
        if not docs:
            text = "I couldn't find that information in the uploaded PDF."
            return {"success": True, "answer": text, "pages": [], "html": f"<h3>Answer</h3><p>{html.escape(text)}</p>"}
        context_parts = []
        source_pages = []
        for doc in docs:
            page = doc.metadata.get("page")
            if page is not None:
                page_number = int(page) + 1
                source_pages.append(page_number)
                context_parts.append(f"[Page {page_number}]\n{doc.page_content}")
            else:
                context_parts.append(doc.page_content)
        context = "\n\n".join(context_parts)
        result = str(chain.invoke({"context": context, "question": question})).strip()
        unique_pages = sorted(set(source_pages))
        source = ", ".join(map(str, unique_pages))
        source_html = f"<div class='source'><strong>Source pages:</strong> {source}</div>" if source else ""
        safe = html.escape(result).replace("\n", "<br>")
        return {"success": True, "answer": result, "pages": unique_pages, "html": f"<h3>Answer</h3><p>{safe}</p>{source_html}"}
    except Exception as exc:
        logger.exception("Question failed")
        return JSONResponse(status_code=500, content={"success": False, "error": f"RAG request failed: {type(exc).__name__}: {exc}"})


@app.post("/clear")
async def clear():
    global vector_store, pdf_name, page_count, chunk_count
    vector_store = None
    pdf_name = None
    page_count = 0
    chunk_count = 0
    return {"success": True, "message": "Knowledge base cleared."}


@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "RAG PDF Agent", "pdf_ready": vector_store is not None, "pdf_name": pdf_name, "pages": page_count, "chunks": chunk_count, "model": MODEL}


@app.get("/api-info")
async def api_info():
    return {"upload": "POST /upload-pdf", "ask": "POST /ask", "clear": "POST /clear", "health": "GET /health", "playground": "/agent/playground/", "docs": "/docs"}


def playground(question: str) -> str:
    question = str(question or "").strip()
    if not question:
        return "Please enter a question."
    if vector_store is None:
        return "Please upload a PDF through the main RAG Agent page first: /"
    try:
        docs = retrieve(question)
        if not docs:
            return "I couldn't find that information in the uploaded PDF."
        parts = []
        for doc in docs:
            page = doc.metadata.get("page")
            prefix = f"[Page {int(page) + 1}]\n" if page is not None else ""
            parts.append(prefix + doc.page_content)
        context = "\n\n".join(parts)
        return str(chain.invoke({"context": context, "question": question})).strip()
    except Exception as exc:
        logger.exception("Playground failed")
        return f"RAG request failed: {type(exc).__name__}: {exc}"


add_routes(app, RunnableLambda(playground), path="/agent", input_type=str, output_type=str)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
