from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import sqlite3
from ocr import extract_text
from llm import generate_summary, answer_query
from utils import preprocess_image
from pdf2image import convert_from_path

app = FastAPI()

# Database setup
DB_PATH = "documents.db"
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS documents
                 (id INTEGER PRIMARY KEY, filename TEXT, text TEXT, summary TEXT)''')
    conn.commit()
    conn.close()

init_db()

# Ensure pdfs directory exists
os.makedirs("pdfs", exist_ok=True)

# Upload and process PDF
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    pdf_path = f"pdfs/{file.filename}"
    with open(pdf_path, "wb") as f:
        f.write(await file.read())
    
    # Convert PDF to images and preprocess
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        preprocessed = preprocess_image(image)
        text += extract_text(preprocessed) + "\n"
    
    # Generate summary
    summary = generate_summary(text)
    
    # Store in database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO documents (filename, text, summary) VALUES (?, ?, ?)",
              (file.filename, text, summary))
    conn.commit()
    conn.close()
    
    return JSONResponse({"filename": file.filename, "summary": summary})

# Query endpoint
@app.post("/query/")
async def query_document(data: dict):
    question = data.get("question")
    filename = data.get("filename")
    
    if not question or not filename:
        raise HTTPException(status_code=400, detail="Question and filename required")
    
    # Retrieve document text
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT text FROM documents WHERE filename = ?", (filename,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        raise HTTPException(status_code=404, detail="Document not found")
    
    text = result[0]
    answer = answer_query(question, text)
    return JSONResponse({"answer": answer})