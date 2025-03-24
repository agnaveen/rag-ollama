# rag-ollama

### Run ollama
Download ollama and install in your machine then open your command prompt and run this,
```bash
ollama serve
ollama run llama3
```

### Running RAG application
```bash

python3 -m venv rag
source rag/bin/activate
pip install flask
pip install sentence-transformers
pip install faiss-cpu
pip install PyMuPDF
pip install frontend
pip install fitz

python3 rag.py

```
