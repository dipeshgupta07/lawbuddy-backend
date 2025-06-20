# main.py

import os
import pickle
import faiss
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from groq import Groq

# Load API key from .env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (for mobile/web frontend access)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load vector index and metadata
index = faiss.read_index("vector_index.faiss")
with open("metadata.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
sources = data["sources"]

# Load sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Function to query Groq's model
def query_groq_model(prompt, context):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful AI legal assistant."},
            {"role": "user", "content": f"{context}\n\nQ: {prompt}\nA:"}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content


# API endpoint for asking legal questions
@app.post("/ask")
async def ask_question(request: Request):
    body = await request.json()
    query = body.get("query", "").strip()

    if not query:
        return {"error": "Query cannot be empty."}

    # Step 1: Embed the query
    query_embedding = model.encode([query])
    D, I = index.search(np.array(query_embedding).astype(np.float32), k=5)

    # Step 2: Retrieve top-k chunks
    context = "\n\n".join([chunks[i] for i in I[0]])

    # Step 3: Generate response from Groq
    answer = query_groq_model(query, context)

    return {"answer": answer}
