from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from semantic_matcher import match_prompt

app = FastAPI()

# Optional: allow requests from any origin (customize this for security in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or ["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/semantic_route")
async def semantic_route(request: Request):
    try:
        data = await request.json()
        prompt = data.get("prompt", "")
        session_id = data.get("session_id", None)
        result = match_prompt(prompt)
        return JSONResponse(content={"session_id": session_id, **result})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/privacy.html")
def serve_privacy_policy():
    return FileResponse("privacy.html", media_type="text/html")

@app.get("/health")
def health_check():
    return {"status": "ok"}
