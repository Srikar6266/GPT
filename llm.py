import requests
import json

# Ollama API endpoint
OLLAMA_API = "http://localhost:11434/api/generate"

def generate_summary(text):
    max_input_length = 4096
    chunks = chunk_text(text, max_input_length)
    summary = ""
    for chunk in chunks:
        payload = {
            "model": "llama3.1:70b",
            "prompt": f"Summarize the following text in 50-150 words:\n{chunk}",
            "stream": False
        }
        response = requests.post(OLLAMA_API, json=payload)
        if response.status_code == 200:
            summary += json.loads(response.text)["response"] + " "
        else:
            raise Exception("Ollama summarization failed")
    return summary.strip()

def answer_query(question, context):
    payload = {
        "model": "llama3.1:70b",
        "prompt": f"Context: {context}\nQuestion: {question}\nAnswer concisely.",
        "stream": False
    }
    response = requests.post(OLLAMA_API, json=payload)
    if response.status_code == 200:
        return json.loads(response.text)["response"]
    else:
        raise Exception("Ollama query failed")

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