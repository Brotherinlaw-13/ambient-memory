"""
Improved FastAPI server combining Phase 3 functionality with Phase 5 API design.

Provides clean HTTP endpoints with better request models, context support,
and temporal features while preserving the working search algorithm.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .search_improved import ImprovedHybridSearcher, SearchConfig
from .chunking import ConversationChunker, Chunk
from .collections import CollectionManager


# Pydantic models for API requests/responses (improved from Phase 5)
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Search query")
    collections: Optional[List[str]] = Field(None, description="Collections to search (null = all)")
    limit: int = Field(10, ge=1, le=100, description="Maximum number of results")
    include_scores: bool = Field(True, description="Include scoring details")
    context: Optional[List[str]] = Field(None, description="Surrounding messages for context expansion")
    query_time: Optional[str] = Field(None, description="ISO timestamp for temporal boosting")


class QueryResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]]
    total_collections: int
    execution_time_ms: int
    config_applied: Dict[str, Any]


class ConfigRequest(BaseModel):
    """Request to update search configuration."""
    semantic_weight: Optional[float] = Field(None, ge=0.0, le=1.0)
    keyword_weight: Optional[float] = Field(None, ge=0.0, le=1.0) 
    min_relevance_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    enable_query_filtering: Optional[bool] = None
    enable_deduplication: Optional[bool] = None
    adaptive_result_gap: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_results: Optional[int] = Field(None, ge=1, le=100)


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Text to ingest")
    source: str = Field(..., description="Source identifier")
    collection: Optional[str] = Field(None, description="Target collection (null = auto-classify)")
    chunk_size: Optional[int] = Field(None, ge=100, le=3000, description="Custom chunk size")
    timestamp: Optional[str] = Field(None, description="ISO timestamp for the content")


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
    embedding_server_url: str = "http://localhost:9876",
    default_config: Optional[SearchConfig] = None
) -> FastAPI:
    """Create FastAPI application with improved ambient memory endpoints."""
    
    app = FastAPI(
        title="Ambient Memory API - Improved",
        description="Hybrid search with Phase 3 algorithm + Phase 5 structure + Phase 6 improvements",
        version="0.2.0"
    )
    
    # Initialise components
    searcher = ImprovedHybridSearcher(
        chroma_path=chroma_path,
        embedding_server_url=embedding_server_url,
        config=default_config
    )
    
    chunker = ConversationChunker()
    
    collection_manager = CollectionManager(
        chroma_path=chroma_path,
        embedding_server_url=embedding_server_url
    )
    
    # Ensure default collections exist
    collection_manager.ensure_default_collections()
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint with configuration info."""
        return {
            "status": "ok",
            "timestamp": datetime.utcnow().isoformat(),
            "embedding_server": searcher.use_embedding_server,
            "collections": len(collection_manager.list_collections()),
            "version": "0.2.0",
            "features": [
                "phase3_algorithm",
                "phase5_structure", 
                "phase6_improvements",
                "configurable_thresholds",
                "query_filtering",
                "result_deduplication"
            ],
            "current_config": {
                "semantic_weight": searcher.config.semantic_weight,
                "keyword_weight": searcher.config.keyword_weight,
                "min_relevance_threshold": searcher.config.min_relevance_threshold,
                "enable_query_filtering": searcher.config.enable_query_filtering,
                "enable_deduplication": searcher.config.enable_deduplication
            }
        }
    
    @app.post("/query", response_model=QueryResponse)
    async def query_memory(request: QueryRequest):
        """Search across memory collections with improved features."""
        start_time = datetime.utcnow()
        
        try:
            # Get available collections
            available_collections = [c["name"] for c in collection_manager.list_collections() if c["count"] > 0]
            
            if not available_collections:
                return QueryResponse(
                    query=request.query,
                    results=[],
                    total_collections=0,
                    execution_time_ms=0,
                    config_applied={}
                )
            
            # Determine collections to search
            search_collections = request.collections or available_collections
            search_collections = [c for c in search_collections if c in available_collections]
            
            if not search_collections:
                raise HTTPException(
                    status_code=400,
                    detail=f"None of the requested collections exist or contain data"
                )
            
            # Parse query time if provided
            query_time = None
            if request.query_time:
                try:
                    query_time = datetime.fromisoformat(request.query_time.replace('Z', '+00:00'))
                except ValueError:
                    pass  # Ignore invalid timestamps
            
            # Perform search with improved searcher
            results = searcher.search(
                query=request.query,
                collections=search_collections,
                limit=request.limit,
                include_scores=request.include_scores,
                query_time=query_time,
                context=request.context
            )
            
            execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return QueryResponse(
                query=request.query,
                results=results,
                total_collections=len(search_collections),
                execution_time_ms=execution_time,
                config_applied={
                    "semantic_weight": searcher.config.semantic_weight,
                    "keyword_weight": searcher.config.keyword_weight,
                    "min_relevance_threshold": searcher.config.min_relevance_threshold,
                    "query_filtered": searcher.config.enable_query_filtering,
                    "deduplication_applied": searcher.config.enable_deduplication
                }
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/config")
    async def update_config(request: ConfigRequest):
        """Update search configuration dynamically."""
        try:
            # Get current config
            current_config = searcher.config
            
            # Update specified fields
            updates = {}
            if request.semantic_weight is not None:
                current_config.semantic_weight = request.semantic_weight
                updates["semantic_weight"] = request.semantic_weight
            
            if request.keyword_weight is not None:
                current_config.keyword_weight = request.keyword_weight
                updates["keyword_weight"] = request.keyword_weight
            
            if request.min_relevance_threshold is not None:
                current_config.min_relevance_threshold = request.min_relevance_threshold
                updates["min_relevance_threshold"] = request.min_relevance_threshold
            
            if request.enable_query_filtering is not None:
                current_config.enable_query_filtering = request.enable_query_filtering
                updates["enable_query_filtering"] = request.enable_query_filtering
            
            if request.enable_deduplication is not None:
                current_config.enable_deduplication = request.enable_deduplication
                updates["enable_deduplication"] = request.enable_deduplication
            
            if request.adaptive_result_gap is not None:
                current_config.adaptive_result_gap = request.adaptive_result_gap
                updates["adaptive_result_gap"] = request.adaptive_result_gap
            
            if request.max_results is not None:
                current_config.max_results = request.max_results
                updates["max_results"] = request.max_results
            
            # Validate weights sum to reasonable range
            total_weight = current_config.semantic_weight + current_config.keyword_weight
            if total_weight > 1.2 or total_weight < 0.8:
                raise HTTPException(
                    status_code=400,
                    detail=f"Semantic and keyword weights should sum to ~1.0, got {total_weight}"
                )
            
            return {
                "status": "config_updated",
                "updates": updates,
                "current_config": {
                    "semantic_weight": current_config.semantic_weight,
                    "keyword_weight": current_config.keyword_weight,
                    "min_relevance_threshold": current_config.min_relevance_threshold,
                    "enable_query_filtering": current_config.enable_query_filtering,
                    "enable_deduplication": current_config.enable_deduplication,
                    "adaptive_result_gap": current_config.adaptive_result_gap,
                    "max_results": current_config.max_results
                }
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/config")
    async def get_config():
        """Get current search configuration."""
        return {
            "semantic_weight": searcher.config.semantic_weight,
            "keyword_weight": searcher.config.keyword_weight,
            "min_relevance_threshold": searcher.config.min_relevance_threshold,
            "enable_query_filtering": searcher.config.enable_query_filtering,
            "enable_deduplication": searcher.config.enable_deduplication,
            "adaptive_result_gap": searcher.config.adaptive_result_gap,
            "max_results": searcher.config.max_results,
            "enable_temporal_boost": searcher.config.enable_temporal_boost,
            "distance_threshold": searcher.config.distance_threshold
        }
    
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
            
            # Add timestamp to metadata if provided
            if request.timestamp:
                try:
                    timestamp = datetime.fromisoformat(request.timestamp.replace('Z', '+00:00'))
                    for chunk in chunks:
                        if hasattr(chunk, 'metadata'):
                            chunk.metadata['timestamp'] = timestamp.isoformat()
                except ValueError:
                    pass  # Ignore invalid timestamps
            
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
            "feedback_id": str(uuid.uuid4()),
            "config_at_time": {
                "semantic_weight": searcher.config.semantic_weight,
                "keyword_weight": searcher.config.keyword_weight,
                "min_relevance_threshold": searcher.config.min_relevance_threshold
            }
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
    
    # Include all other endpoints from original server.py
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
    
    return app


# Default app instance with Phase 6 improvements enabled
default_config = SearchConfig(
    min_relevance_threshold=0.0,  # Start conservative for coverage
    enable_query_filtering=True,
    enable_deduplication=True,
    adaptive_result_gap=0.0       # Disabled initially
)

app = create_app(default_config=default_config)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server_improved:app", host="127.0.0.1", port=8000, reload=True)