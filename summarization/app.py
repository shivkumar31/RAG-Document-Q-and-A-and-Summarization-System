from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from ingest import ingest_pdf
from rag import ask_question, summarize_document

app = FastAPI(title="RAG")


app.state.vectorstore = None
app.state.chunks = None


class QuestionRequest(BaseModel):
    question: str


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    vectorstore, chunks = ingest_pdf(file)

    app.state.vectorstore = vectorstore
    app.state.chunks = chunks

    return {
        "message": "Document processed successfully.",
        "chunks": len(chunks),
    }


@app.post("/ask")
async def ask(request: QuestionRequest):
    if app.state.vectorstore is None:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF first."
        )

    return ask_question(
        app.state.vectorstore,
        request.question,
    )


@app.post("/summarize")
async def summarize():
    if app.state.chunks is None:
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF first."
        )

    summary = summarize_document(app.state.chunks)

    return {
        "summary": summary
    }