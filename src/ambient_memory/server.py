"""
HTTP API server for Ambient Memory.

Simple REST endpoints for storing and retrieving agent memories.
"""

from fastapi import FastAPI

app = FastAPI(
    title="Ambient Memory",
    description="Open source ambient memory for AI agents",
    version="0.1.0",
)


@app.get("/")
async def root():
    return {"message": "Ambient Memory API v0.1.0"}


@app.post("/memory")
async def store_memory(content: str, topic: str = "general"):
    """Store a memory chunk."""
    # TODO: Implement memory storage with hybrid search indexing
    pass


@app.get("/search")
async def search_memories(query: str, topic: str = None, limit: int = 10):
    """Search memories using hybrid search."""
    # TODO: Implement hybrid search (semantic + keyword/entity matching)
    pass


@app.post("/feedback")
async def feedback(memory_id: str, score: int):
    """Provide feedback on retrieved memory relevance (+1/0/-1)."""
    # TODO: Implement feedback scoring for threshold auto-tuning
    pass