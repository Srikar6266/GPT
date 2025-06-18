import requests
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama API endpoint
OLLAMA_API = "http://localhost:11434/api/generate"

def generate_summary(text):
    max_input_length = 4096
    chunks = chunk_text(text, max_input_length)
    summary = ""
    try:
        for chunk in chunks:
            payload = {
                "model": "llama3.1:70b",
                "prompt": f"""Summarize the following text in 50-150 words, focusing on key details such as location, area, zoning, or legal terms if applicable:\n{chunk}""",
                "stream": False
            }
            response = requests.post(OLLAMA_API, json=payload)
            if response.status_code == 200:
                summary += json.loads(response.text)["response"] + " "
            else:
                logger.error(f"Ollama summarization failed: {response.text}")
                raise Exception("Ollama summarization failed")
        logger.info("Summary generated successfully")
        return summary.strip()
    except Exception as e:
        logger.error(f"Summary generation failed: {e}")
        return "Summary generation failed."

def answer_query(question, context):
    try:
        # Dynamic prompt for flexible question answering
        prompt = f"""You are an expert in real estate document analysis. Using the provided context, answer the question as accurately and concisely as possible. If the question is open-ended, provide a detailed response. If the answer is not in the context, say so clearly.

Context: {context}

Question: {question}

Answer:"""
        payload = {
            "model": "llama3.1:70b",
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(OLLAMA_API, json=payload)
        if response.status_code == 200:
            answer = json.loads(response.text)["response"].strip()
            logger.info(f"Query answered: {question}")
            return answer
        else:
            logger.error(f"Ollama query failed: {response.text}")
            raise Exception("Ollama query failed")
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        return "Query processing failed."

def chunk_text(text, max_length=4096):
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    for word in words:
        current_length += len(word) + 1
        if current_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word) + 1
        else:
            current_chunk.append(word)
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks