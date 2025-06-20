# backend/main.py

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy globals
index = None
chunks = None
sources = None
model = None

def load_dependencies():
    global index, chunks, sources, model
    if index is None:
        index = faiss.read_index("vector_index.faiss")
    if chunks is None or sources is None:
        with open("metadata.pkl", "rb") as f:
            meta = pickle.load(f)
            chunks = meta["chunks"]
            sources = meta["sources"]
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")

@app.post("/ask")
async def ask_question(request: Request):
    load_dependencies()
    body = await request.json()
    query = body.get("query", "")

    # Embed and search
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).astype(np.float32), k=5)
    context = "\n\n".join([chunks[i] for i in I[0]])

    # Use Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "user", "content": f"Answer the question using the context:\n\n{context}\n\nQ: {query}\nA:"}
        ]
    )

    return {"answer": response.choices[0].message.content}
