# Document Intelligence System

A Document Intelligence System built with **LangChain**, **Ollama (Llama 3.1:8B)**, **HuggingFace Embeddings**, **ChromaDB**, and **FastAPI** that enables users to upload a PDF and interact with it through semantic question answering and comprehensive document summarization.

## 🚀 Problem Statement

Understanding large documents such as research papers, technical manuals, reports, or policy documents is both time-consuming and inefficient. Users generally need two key capabilities:

* **Question Answering:** Find a specific piece of information without reading the entire document.
* **Document Summarization:** Obtain a complete and faithful summary of the entire document.

Traditional keyword search struggles with semantic understanding, while using Retrieval-Augmented Generation (RAG) alone for summarization is ineffective because it retrieves only the most relevant chunks rather than processing the complete document.

## 💡 Solution

This project combines two complementary approaches to solve these problems effectively:

### Semantic Question Answering (RAG)

After a PDF is uploaded, the system:

* Extracts text from the document.
* Splits the content into semantic chunks.
* Generates vector embeddings using **all-MiniLM-L6-v2**.
* Stores embeddings in an **in-memory Chroma vector database**.
* Retrieves the most relevant chunks for a user's query.
* Uses **Llama 3.1 (Ollama)** to generate grounded answers with page-level citations.

### Full Document Summarization

To generate a faithful summary of the entire document, the system uses a **Map-Reduce** workflow:

* **Map Phase:** Each document chunk is summarized independently.
* **Reduce Phase:** All chunk summaries are combined into a single comprehensive summary.

Unlike RAG, this approach ensures that every section of the document contributes to the final summary.

## ✨ Features

* Upload and process PDF documents
* Semantic Question Answering using Retrieval-Augmented Generation (RAG)
* Page-level source citations for generated answers
* Full-document Map-Reduce summarization
* In-memory Chroma vector database for fast retrieval
* Local inference using Ollama (Llama 3.1:8B)
* RESTful FastAPI backend

## 🛠️ Tech Stack

* Python
* FastAPI
* LangChain
* Ollama (Llama 3.1:8B)
* HuggingFace Embeddings (`all-MiniLM-L6-v2`)
* ChromaDB
* PyPDF
* RecursiveCharacterTextSplitter

## 🎯 Key Highlights

* Built a semantic document search system using Retrieval-Augmented Generation (RAG).
* Implemented a Map-Reduce summarization pipeline for complete document understanding.
* Leveraged local LLM inference with Ollama to eliminate dependency on external APIs.
* Designed a lightweight, in-memory architecture for efficient single-document processing.
