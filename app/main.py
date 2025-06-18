from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os
import asyncio
from ocr import extract_text_batch
from llm import generate_summary, answer_query
from utils import preprocess_image
from pdf2image import convert_from_path
from database import init_db, store_document, get_document_text
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize DB
@app.on_event("startup")
async def startup_event():
    await init_db()
    os.makedirs("pdfs", exist_ok=True)
    logger.info("App started, DB initialized, pdfs dir ready")

# Upload and process PDF
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        logger.error("Invalid file: not a PDF")
        raise HTTPException(status_code=400, detail="Only PDF files allowed, maccha!")
    
    pdf_path = f"pdfs/{file.filename}"
    try:
        # Save PDF
        with open(pdf_path, "wb") as f:
            f.write(await file.read())
        
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        preprocessed_images = [preprocess_image(image) for image in images]
        
        # Batch extract text
        text = await extract_text_batch(preprocessed_images)
        
        # Generate summary async
        summary = await generate_summary(text)
        
        # Store in DB
        await store_document(file.filename, text, summary)
        logger.info(f"PDF processed: {file.filename}")
        return JSONResponse({
            "filename": file.filename,
            "summary": summary,
            "message": "PDF uploaded and ready to roll, dude!"
        })
    except Exception as e:
        logger.error(f"PDF processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Oops, something broke: {str(e)}")

# Query endpoint
@app.post("/query/")
async def query_document(data: dict):
    question = data.get("question")
    filename = data.get("filename")
    
    if not question or not filename:
        logger.error("Missing question or filename")
        raise HTTPException(status_code=400, detail="Gimme a question and filename, maccha!")
    
    try:
        # Get document text
        text = await get_document_text(filename)
        if not text:
            logger.error(f"Document not found: {filename}")
            raise HTTPException(status_code=404, detail="No such doc, dude!")
        
        # Answer query async
        answer = await answer_query(question, text)
        logger.info(f"Query answered for {filename}: {question}")
        return JSONResponse({
            "question": question,
            "answer": answer,
            "message": "Nailed it, maccha!"
        })
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query crashed: {str(e)}")     