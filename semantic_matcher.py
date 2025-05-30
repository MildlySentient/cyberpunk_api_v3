import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Load embeddings
with open("canonical_embeddings.json", "r", encoding="utf-8") as f:
    VECTORS = json.load(f)

# Prepare matrix and node IDs
NODE_IDS = list(VECTORS.keys())
MATRIX = normalize(np.array([VECTORS[node_id] for node_id in NODE_IDS]))

# Lazy-load transformer model
_model = None

def match_prompt(prompt, threshold=0.5):
    global _model

    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")

    # Encode and normalize input prompt
    embedding = _model.encode([prompt], convert_to_numpy=True)
    embedding = normalize(embedding)[0]

    # Compute cosine similarity
    scores = np.dot(MATRIX, embedding)
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])

    return {
        "node_id": str(NODE_IDS[best_idx]),
        "score": best_score,
        "match": bool(best_score >= threshold)
    }
