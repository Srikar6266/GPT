import httpx
import json
import asyncio
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama API endpoint
OLLAMA_API = "http://localhost:11434/api/generate"

async def generate_summary(text):
    max_input_length = 4096
    chunks = chunk_text(text, max_input_length)
    summary = ""
    async with httpx.AsyncClient(timeout=60.0) as client:
        for chunk in chunks:
            try:
                payload = {
                    "model": "llama3.1:70b",
                    "prompt": f"""Summarize this in 50-150 words, hitting key details like location, area, zoning, or legal terms:\n{chunk}""",
                    "stream": False
                }
                response = await client.post(OLLAMA_API, json=payload)
                response.raise_for_status()
                summary += json.loads(response.text)["response"] + " "
                logger.info("Summary chunk generated")
            except Exception as e:
                logger.error(f"Summary chunk failed: {e}")
                return "Summary failed, maccha!"
    return summary.strip()

async def answer_query(question, context):
    try:
        prompt = f"""You're a real estate doc genius, maccha! Using the context, answer the question sharp and clear. For open-ended stuff, give a solid breakdown. If the answer ain't in the context, say so straight up.

Context: {context}

Question: {question}

Answer:"""
        async with httpx.AsyncClient(timeout=60.0) as client:
            payload = {
                "model": "llama3.1:70b",
                "prompt": prompt,
                "stream": False
            }
            response = await client.post(OLLAMA_API, json=payload)
            response.raise_for_status()
            answer = json.loads(response.text)["response"].strip()
            logger.info(f"Query answered: {question}")
            return answer
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return "Query bombed, dude!"

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