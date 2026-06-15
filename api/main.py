import asyncio
import json

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel

from observability import configure_logging, get_logger, bind_correlation_id, new_correlation_id
from agent.graph import run_query

configure_logging()
app = FastAPI(title="Sheares Hall Policy Copilot", version="0.2.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
log = get_logger("api.main")

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []
    stream: bool = True

class FeedbackRequest(BaseModel):
    correlation_id: str
    rating: int   # 1 = thumbs up, -1 = thumbs down
    comment: str = ""

@app.get("/health")
async def health():
    return {"status": "ok", "service": "sheares-policy-copilot"}

@app.post("/chat")
async def chat(req: ChatRequest):
    cid = bind_correlation_id()
    log.info("api.chat.received", query=req.message[:80], stream=req.stream, correlation_id=cid)

    if req.stream:
        async def generate():
            result = await run_query(req.message, req.history, correlation_id=cid)
            # Stream the answer word by word for demo effect
            words = result["answer"].split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words)-1 else "")
                yield json.dumps({"type": "token", "content": chunk, "correlation_id": cid})
                await asyncio.sleep(0.02)
            # Send metadata at end
            yield json.dumps({
                "type": "done",
                "citations": result.get("citations", []),
                "route": result.get("route", "fast"),
                "is_grounded": result.get("is_grounded", True),
                "tool_results": result.get("tool_results", []),
                "correlation_id": cid
            })

        return EventSourceResponse(generate())
    else:
        result = await run_query(req.message, req.history, correlation_id=cid)
        log.info("api.chat.done", correlation_id=cid, route=result.get("route"))
        return {**result, "correlation_id": cid}

@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    log.info("api.feedback", **req.model_dump())
    # In production: write to a feedback store / Langfuse score
    return {"status": "recorded"}
