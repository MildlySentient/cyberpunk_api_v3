from fastapi import FastAPI, Request
from semantic_matcher import match_prompt

app = FastAPI()

@app.post("/semantic_route")
async def semantic_route(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    session_id = data.get("session_id", None)
    result = match_prompt(prompt)
    return {"session_id": session_id, **result}
