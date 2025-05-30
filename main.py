from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from semantic_matcher import match_prompt

app = FastAPI()

# Allow CORS (for development; restrict origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your frontend domain in production
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

        if not prompt:
            return JSONResponse(
                content={"error": "Missing 'prompt' field."},
                status_code=400
            )

        result = match_prompt(prompt)
        return JSONResponse(content={"session_id": session_id, **result})

    except Exception as e:
        return JSONResponse(
            content={"error": f"Internal server error: {str(e)}"},
            status_code=500
        )

@app.get("/privacy.html")
def serve_privacy_policy():
    return FileResponse("privacy.html", media_type="text/html")

@app.get("/health")
def health_check():
    return {"status": "ok"}
