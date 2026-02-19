#!/usr/bin/env python3
"""
Simplified Ambient Memory Server

A basic FastAPI server that provides the ambient memory API
using the existing embedding server as the backend.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional
import httpx
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# Request/Response models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Search query")
    collections: Optional[List[str]] = Field(None, description="Collections to search (ignored for now)")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    include_scores: bool = Field(True, description="Include scoring details")


class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_collections: int
    execution_time_ms: int


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    embedding_server: str
    version: str


def create_app(embedding_server_url: str = "http://localhost:9876") -> FastAPI:
    """Create the simplified ambient memory FastAPI app."""
    
    app = FastAPI(
        title="Ambient Memory API (Simplified)",
        description="Hybrid search and memory management for AI agents",
        version="0.1.0-simplified"
    )
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint."""
        # Test embedding server
        embedding_status = "offline"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{embedding_server_url}/health", timeout=5.0)
                if response.status_code == 200:
                    embedding_status = "online"
        except Exception:
            pass
        
        return HealthResponse(
            status="ok",
            timestamp=datetime.utcnow().isoformat(),
            embedding_server=embedding_status,
            version="0.1.0-simplified"
        )
    
    @app.post("/query", response_model=QueryResponse)
    async def query_memory(request: QueryRequest):
        """Search memory via existing embedding server."""
        start_time = datetime.utcnow()
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{embedding_server_url}/search",
                    params={
                        "q": request.query,
                        "n": request.limit
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                data = response.json()
            
            # Transform results to match our API format
            results = []
            for result in data.get("results", []):
                transformed_result = {
                    "text": result.get("snippet", ""),
                    "source": result.get("source", ""),
                    "collection": result.get("collection", "memory_general"),
                    "final_score": result.get("final_score", 0),
                }
                
                if request.include_scores:
                    transformed_result.update({
                        "similarity": result.get("similarity", 0),
                        "keyword_score": result.get("keyword_score", 0),
                    })
                
                results.append(transformed_result)
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return QueryResponse(
                query=request.query,
                results=results,
                total_collections=1,  # Simplified - we only have one collection
                execution_time_ms=execution_time
            )
            
        except httpx.HTTPError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Embedding server error: {str(e)}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Internal error: {str(e)}"
            )
    
    @app.get("/")
    async def root():
        """Root endpoint with basic info."""
        return {
            "name": "Ambient Memory API",
            "version": "0.1.0-simplified",
            "status": "running",
            "docs": "/docs",
            "health": "/health"
        }
    
    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    print("🧠 Starting Ambient Memory Server (Simplified)")
    print("📡 Backend: Existing embedding server at localhost:9876")
    print("🌐 API will be available at: http://localhost:8000")
    print("📖 Docs available at: http://localhost:8000/docs")
    uvicorn.run("simple_server:app", host="127.0.0.1", port=8000, reload=True)