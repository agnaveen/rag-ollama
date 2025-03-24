import fitz  # PyMuPDF
import faiss
import numpy as np
import requests
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

# --- Config ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"  # or mistral, gemma, etc.

# --- Initialize ---
app = Flask(__name__, static_folder=None)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = []
index = None


# --- Utils ---

def load_pdf_text(file_obj):
    """Extract text from uploaded PDF (in-memory)."""
    doc = fitz.open(stream=file_obj.read(), filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks for embedding."""
    words = text.split()
    return [
        " ".join(words[i:i+chunk_size])
        for i in range(0, len(words), chunk_size - overlap)
    ]

def embed_chunks(chunks, model):
    """Generate dense vector embeddings for each text chunk."""
    return np.array(model.encode(chunks)).astype("float32")

def build_faiss_index(embeddings):
    """Create and populate FAISS index from embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def query_rag(question, chunks, index, embed_model, top_k=3):
    """Perform RAG query using vector search and Ollama."""
    q_emb = embed_model.encode([question]).astype("float32")
    D, I = index.search(q_emb, top_k)
    context = "\n\n".join([chunks[i] for i in I[0]])
    prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    return call_ollama(prompt)

def call_ollama(prompt):
    """Send prompt to local Ollama API."""
    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    })
    return response.json().get("response")


# --- Flask Routes ---

@app.route("/load_pdf", methods=["POST"])
def load_pdf():
    global chunks, index
    if 'pdf' not in request.files:
        return jsonify({"error": "Missing PDF file"}), 400

    file = request.files['pdf']
    text = load_pdf_text(file)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks, embed_model)
    index = build_faiss_index(embeddings)

    return jsonify({
        "status": "PDF loaded and indexed successfully",
        "chunks": len(chunks)
    })


@app.route("/ask", methods=["POST"])
def ask():
    if index is None:
        return jsonify({"error": "No PDF loaded yet"}), 400

    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    answer = query_rag(question, chunks, index, embed_model)
    return jsonify({"answer": answer})


# --- Run Server ---
if __name__ == "__main__":
    app.run(port=5000)