"""
FastAPI server for the ambient memory system.

Provides HTTP endpoints for querying, ingesting, and managing memory collections.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .search import HybridSearcher
from .chunking import ConversationChunker, Chunk
from .collections import CollectionManager
from .ingest import IngestPipeline, IngestConfig, classify
from .quarantine import QuarantineTracker


# Pydantic models for API requests/responses
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Search query")
    collections: Optional[List[str]] = Field(None, description="Collections to search (null = all)")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    include_scores: bool = Field(True, description="Include scoring details")


class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_collections: int
    execution_time_ms: int


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Text to ingest")
    source: str = Field(..., description="Source identifier")
    collection: Optional[str] = Field(None, description="Target collection (null = auto-classify)")
    chunk_size: Optional[int] = Field(None, ge=100, le=3000, description="Custom chunk size")


class IngestResponse(BaseModel):
    ingestion_stats: Dict[str, int]
    total_chunks: int
    collections_used: List[str]


class FeedbackRequest(BaseModel):
    query: str = Field(..., description="Original query")
    result_text: str = Field(..., description="Result text that was rated")
    score: int = Field(..., ge=-1, le=1, description="Feedback score: -1 (noise), 0 (neutral), 1 (helpful)")
    collection: Optional[str] = Field(None, description="Collection the result came from")


class CollectionInfo(BaseModel):
    name: str
    description: str
    count: int
    sources: List[str]
    speakers: List[str]


# Feedback storage
feedback_file = Path(".ambient-memory-feedback.jsonl")


def log_feedback(feedback: Dict[str, Any]):
    """Log feedback to JSONL file."""
    try:
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback) + "\n")
    except Exception as e:
        print(f"Error logging feedback: {e}")


def create_app(
    chroma_path: str = ".chromadb",
    embedding_server_url: str = "http://localhost:9876"
) -> FastAPI:
    """Create FastAPI application with ambient memory endpoints."""
    
    app = FastAPI(
        title="Ambient Memory API",
        description="Hybrid search and memory management for AI agents",
        version="0.1.0"
    )
    
    # Initialise components
    searcher = HybridSearcher(
        chroma_path=chroma_path,
        embedding_server_url=embedding_server_url
    )
    
    chunker = ConversationChunker()
    
    collection_manager = CollectionManager(
        chroma_path=chroma_path,
        embedding_server_url=embedding_server_url
    )
    
    # Initialise ingest pipeline for auto-classification
    ingest_config = IngestConfig()
    ingest_pipeline = IngestPipeline(
        chroma_client=searcher.chroma_client,
        config=ingest_config
    )
    
    # Initialise quarantine tracker
    quarantine_tracker = QuarantineTracker()
    
    # Ensure default collections exist
    collection_manager.ensure_default_collections()
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "embedding_server": searcher.use_embedding_server,
            "collections": len(collection_manager.list_collections())
        }

    # ── Compatibility endpoints (old embedding-server.py format) ──

    @app.get("/search")
    async def compat_search(q: str, n: int = 5, collection: Optional[str] = None):
        """
        Compatibility endpoint for the old embedding server.
        Maps GET /search?q=...&n=... to the new POST /query format.
        Returns results with 'snippet' and 'similarity' keys for backward compat.
        """
        try:
            available = [c["name"] for c in collection_manager.list_collections() if c["count"] > 0]
            if not available:
                return {"query": q, "results": []}

            search_cols = [collection] if collection and collection in available else available
            results = searcher.search(query=q, collections=search_cols, limit=n, include_scores=True)

            # Track quarantine accesses
            for r in results:
                if r.get("collection") == ingest_config.quarantine_collection:
                    # Extract memory_id from result if available
                    memory_id = r.get("memory_id") or r.get("id")
                    if memory_id:
                        quarantine_tracker.record_access(memory_id)

            # Map to old format
            compat_results = []
            for r in results:
                compat_results.append({
                    "snippet": r.get("text", r.get("snippet", "")),
                    "source": r.get("source", ""),
                    "collection": r.get("collection", ""),
                    "similarity": r.get("semantic_similarity", r.get("final_score", 0.0)),
                    "keyword_score": r.get("keyword_score", 0.0),
                    "final_score": r.get("final_score", 0.0),
                })

            return {"query": q, "results": compat_results}
        except Exception as e:
            return {"query": q, "results": [], "error": str(e)}
    
    @app.post("/query", response_model=QueryResponse)
    async def query_memory(request: QueryRequest):
        """Search across memory collections."""
        start_time = datetime.utcnow()
        
        try:
            # Get available collections
            available_collections = [c["name"] for c in collection_manager.list_collections() if c["count"] > 0]
            
            if not available_collections:
                return QueryResponse(
                    query=request.query,
                    results=[],
                    total_collections=0,
                    execution_time_ms=0
                )
            
            # Determine collections to search
            search_collections = request.collections or available_collections
            search_collections = [c for c in search_collections if c in available_collections]
            
            if not search_collections:
                raise HTTPException(
                    status_code=400,
                    detail=f"None of the requested collections exist or contain data"
                )
            
            # Perform search
            results = searcher.search(
                query=request.query,
                collections=search_collections,
                limit=request.limit,
                include_scores=request.include_scores
            )
            
            # Track quarantine accesses
            for r in results:
                if r.get("collection") == ingest_config.quarantine_collection:
                    # Extract memory_id from result if available
                    memory_id = r.get("memory_id") or r.get("id")
                    if memory_id:
                        quarantine_tracker.record_access(memory_id)
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return QueryResponse(
                query=request.query,
                results=results,
                total_collections=len(search_collections),
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/ingest/compat")
    async def compat_ingest(request: dict):
        """
        Compatibility endpoint for the memory-ingest plugin.
        Accepts {content, collection, metadata} format from the OpenClaw plugin.
        Stores directly into ChromaDB without chunking.
        """
        try:
            content = request.get("content", "")
            collection_name = request.get("collection")
            metadata = request.get("metadata", {})
            
            # Auto-classify if no collection specified
            if not collection_name:
                collection_name = classify(content)

            if not content or len(content) < 10:
                raise HTTPException(status_code=400, detail="Content too short")

            import hashlib
            memory_id = hashlib.sha256(
                f"{content[:200]}:{metadata.get('timestamp', '')}".encode()
            ).hexdigest()[:16]

            # Get or create collection and store directly
            coll = searcher.chroma_client.get_or_create_collection(name=collection_name)
            coll.upsert(
                ids=[memory_id],
                documents=[content],
                metadatas=[{k: str(v) for k, v in metadata.items()}] if metadata else None,
            )

            return {"id": memory_id, "collection": collection_name, "chunks": 1, "metadata": metadata}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest_text(request: IngestRequest):
        """Ingest text into memory collections."""
        try:
            # Configure chunker if custom chunk size provided
            if request.chunk_size:
                chunker.target_chunk_size = request.chunk_size
                chunker.max_chunk_size = int(request.chunk_size * 1.5)
            
            # Chunk the text
            chunks = chunker.chunk_text(request.text, request.source)
            
            if not chunks:
                raise HTTPException(
                    status_code=400,
                    detail="No valid chunks could be created from the provided text"
                )
            
            # Ingest chunks
            ingestion_stats = collection_manager.ingest_chunks(
                chunks=chunks,
                source=request.source,
                auto_classify=request.collection is None,
                target_collection=request.collection
            )
            
            return IngestResponse(
                ingestion_stats=ingestion_stats,
                total_chunks=len(chunks),
                collections_used=list(ingestion_stats.keys())
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/feedback")
    async def submit_feedback(request: FeedbackRequest, background_tasks: BackgroundTasks):
        """Submit feedback on search results."""
        feedback_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": request.query,
            "result_text": request.result_text[:200],  # Truncate for privacy
            "score": request.score,
            "collection": request.collection,
            "feedback_id": str(uuid.uuid4())
        }
        
        # Log feedback in background
        background_tasks.add_task(log_feedback, feedback_data)
        
        return {"status": "feedback_recorded", "feedback_id": feedback_data["feedback_id"]}
    
    @app.get("/collections", response_model=List[CollectionInfo])
    async def list_collections():
        """List all collections with their statistics."""
        try:
            collections = []
            for col_data in collection_manager.list_collections():
                # Get detailed stats
                stats = collection_manager.get_collection_stats(col_data["name"])
                if stats:
                    collections.append(CollectionInfo(
                        name=stats["name"],
                        description=col_data["metadata"].get("description", ""),
                        count=stats["count"],
                        sources=stats["sources"][:10],  # Limit for response size
                        speakers=stats["speakers"][:10]  # Limit for response size
                    ))
            
            return collections
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/collections/{collection_name}")
    async def create_collection(collection_name: str, description: str = ""):
        """Create a new collection."""
        try:
            success = collection_manager.create_collection(collection_name, description)
            if success:
                return {"status": "created", "name": collection_name}
            else:
                raise HTTPException(
                    status_code=409,
                    detail=f"Collection '{collection_name}' already exists"
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.delete("/collections/{collection_name}")
    async def delete_collection(collection_name: str):
        """Delete a collection."""
        try:
            success = collection_manager.delete_collection(collection_name)
            if success:
                return {"status": "deleted", "name": collection_name}
            else:
                raise HTTPException(
                    status_code=404,
                    detail=f"Collection '{collection_name}' not found"
                )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/ingest/file")
    async def ingest_file(
        file: UploadFile = File(...),
        collection: Optional[str] = None,
        chunk_size: Optional[int] = None
    ):
        """Ingest a text file into memory collections."""
        try:
            # Read file content
            content = await file.read()
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="File must be UTF-8 encoded text"
                )
            
            if not text.strip():
                raise HTTPException(
                    status_code=400,
                    detail="File appears to be empty"
                )
            
            # Configure chunker if custom chunk size provided
            if chunk_size:
                chunker.target_chunk_size = chunk_size
                chunker.max_chunk_size = int(chunk_size * 1.5)
            
            # Chunk the text
            chunks = chunker.chunk_text(text, file.filename or "uploaded_file")
            
            if not chunks:
                raise HTTPException(
                    status_code=400,
                    detail="No valid chunks could be created from the file"
                )
            
            # Ingest chunks
            ingestion_stats = collection_manager.ingest_chunks(
                chunks=chunks,
                source=file.filename or "uploaded_file",
                auto_classify=collection is None,
                target_collection=collection
            )
            
            return IngestResponse(
                ingestion_stats=ingestion_stats,
                total_chunks=len(chunks),
                collections_used=list(ingestion_stats.keys())
            )
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/quarantine/candidates")
    async def get_quarantine_candidates():
        """Get list of quarantine memories eligible for promotion."""
        try:
            candidates = quarantine_tracker.get_promotion_candidates()
            return {
                "candidates": candidates,
                "total_count": len(candidates),
                "criteria": {
                    "min_access_count": 3,
                    "min_hours_since_first": 24,
                    "exclude_flagged": True
                }
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return app


# Default app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)