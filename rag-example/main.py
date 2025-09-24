import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
 
# -----------------------------
# Step 1: Knowledge Base
# -----------------------------
documents = [
    "The capital of France is Paris.",
    "Python is a programming language widely used for AI.",
    "The Great Wall of China is visible from space is a myth.",
    "Form 1120 is used by U.S. corporations to file income tax returns."
]
 
# -----------------------------
# Step 2: Encode documents with embeddings
# -----------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(documents)
print(embeddings.shape)# Should be (4, 384)
print("embeddings:",embeddings)  # Print embedding for the first document
# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))
 
# -----------------------------
# Step 3: User query
# -----------------------------
query = "What is Form 1120 used for?"
query_embedding = model.encode([query])
 
# Retrieve top-k results
k = 2
D, I = index.search(np.array(query_embedding), k)
retrieved_docs = [documents[i] for i in I[0]]
 
print("Retrieved context:", retrieved_docs)
 
# -----------------------------
# Step 4: Use Groq LLM with context
# -----------------------------
client = Groq(api_key="")
 
context = "\n".join(retrieved_docs)
prompt = f"""
Answer the question using the context below:
 
Context: {context}
 
Question: {query}
Answer:
"""
 
chat_completion = client.chat.completions.create(
    messages=[{"role": "user", "content": prompt}],
    model="openai/gpt-oss-20b"   # you can also try "mixtral-8x7b-32768"
)
 
print("LLM Answer:", chat_completion.choices[0].message.content)