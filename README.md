# 📚 RAG PDF Agent — Knowledge Transfer Assistant

> A Retrieval-Augmented Generation (RAG) agent that parses PDF documents, creates semantic embeddings, retrieves relevant information, and answers questions using the uploaded document as its knowledge source.

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C7C54?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Google%20Gemini-LLM-4285F4?style=for-the-badge&logo=google)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-FF6B35?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge&logo=render&logoColor=black)

---

# 🌟 Overview

The **RAG PDF Agent** is an AI-powered Knowledge Transfer assistant designed to make information inside PDF documents searchable and conversational.

Users can upload a PDF document and ask questions about its contents using natural language.

Instead of directly asking an LLM to answer from general knowledge, the application first retrieves relevant information from the uploaded document and then provides that context to Gemini.

This creates a document-grounded question-answering workflow.

---

# 🎯 Purpose

The agent is designed for **Knowledge Transfer (KT)** workflows where important information may be stored inside documents such as:

- 📄 Project documentation
- 📚 Study material
- 📝 Technical documentation
- 📋 Requirement documents
- 💼 Resumes
- 🏢 Company documentation
- 📖 Training material
- 🧑‍💻 Project reports

The goal is to allow users to interact with these documents through natural language instead of manually searching through pages.

---

# ✨ Key Features

## 📄 PDF Upload

Users can upload a PDF directly through the web interface.

The application validates:

- File type
- File size
- File availability

The current application supports PDFs up to **10 MB**.

---

## 🔍 PDF Parsing

Uploaded PDFs are processed using LangChain's PDF loading functionality.

The document is converted into structured page-level documents.

```text
PDF
 ↓
PDF Parser
 ↓
Page Documents
