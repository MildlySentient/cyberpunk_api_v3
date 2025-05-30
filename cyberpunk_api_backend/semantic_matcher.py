import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Load vectors
with open("canonical_embeddings.json") as f:
    VECTORS = json.load(f)

# Pre-normalized embedding matrix and ids
NODE_IDS = list(VECTORS.keys())
MATRIX = normalize(np.array([VECTORS[n] for n in NODE_IDS]))

# Load model
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def match_prompt(prompt, threshold=0.5):
    emb = MODEL.encode([prompt], convert_to_numpy=True)
    emb = normalize(emb)[0]
    scores = np.dot(MATRIX, emb)
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    return {
        "node_id": NODE_IDS[best_idx],
        "score": float(best_score),
        "match": best_score >= threshold
    }
