"""
Topic-based collection management for ChromaDB.

Organises memory by topics/contexts to improve search precision and enable
targeted querying across different domains of knowledge.
"""

import uuid
from typing import List, Dict, Optional, Any, Set
from pathlib import Path
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import httpx

from .chunking import Chunk


class CollectionManager:
    """
    Manages ChromaDB collections organised by topic/context.
    
    Features:
    - Create and manage collections per topic (work, personal, projects, etc.)
    - Smart topic assignment based on content analysis
    - Collection metadata and statistics
    - Batch operations for ingestion
    """
    
    def __init__(
        self,
        chroma_path: str = ".chromadb",
        embedding_server_url: str = "http://localhost:9876"
    ):
        """
        Initialise collection manager.
        
        Args:
            chroma_path: Path to ChromaDB storage directory
            embedding_server_url: URL to embedding server for embeddings
        """
        self.chroma_path = Path(chroma_path)
        self.embedding_server_url = embedding_server_url
        
        # Initialise ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.chroma_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Check if embedding server is available
        self.use_embedding_server = self._check_embedding_server()
        
        # Define default collections and their purposes
        self.default_collections = {
            "memory_work": "Work-related discussions, meetings, and professional content",
            "memory_projects": "Specific project discussions, development, and progress",
            "memory_personal": "Personal conversations, interests, and non-work topics",
            "memory_infrastructure": "Technical setup, deployment, servers, and tooling",
            "memory_general": "General conversations that don't fit other categories",
        }
        
        # Topic classification rules (simple keyword-based for now)
        self.topic_rules = {
            "memory_work": [
                "meeting", "standup", "client", "deadline", "business", "company", 
                "revenue", "hire", "recruiting", "interview", "job", "salary",
                "invoice", "payment", "contract", "seo", "marketing"
            ],
            "memory_projects": [
                "darwin", "stitch", "hire space", "root juice", "book social",
                "pocket guide", "railway", "github", "repository", "deploy",
                "feature", "bug", "release", "version", "api"
            ],
            "memory_infrastructure": [
                "server", "database", "chromadb", "vector", "embedding", "api",
                "docker", "vercel", "hosting", "domain", "ssl", "backup",
                "monitoring", "logging", "error", "performance", "scalability"
            ],
            "memory_personal": [
                "family", "friend", "weekend", "holiday", "travel", "food",
                "movie", "book", "game", "sport", "hobby", "health", "weather"
            ]
        }
    
    def _check_embedding_server(self) -> bool:
        """Check if embedding server is available."""
        try:
            response = httpx.get(f"{self.embedding_server_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
    
    def _get_embedding(self, text: str, prefix: str = "passage") -> List[float]:
        """Get embedding for text, either from server or locally."""
        if self.use_embedding_server:
            try:
                response = httpx.post(
                    f"{self.embedding_server_url}/embed",
                    json={"text": text, "prefix": prefix},
                    timeout=10.0
                )
                response.raise_for_status()
                return response.json()["embedding"]
            except Exception:
                # Fall back to local if server fails
                return self._get_embedding_local(text, prefix)
        else:
            return self._get_embedding_local(text, prefix)
    
    def _get_embedding_local(self, text: str, prefix: str = "passage") -> List[float]:
        """Get embedding using local model (fallback)."""
        # Import here to avoid loading model if server is available
        from sentence_transformers import SentenceTransformer
        if not hasattr(self, '_local_model'):
            self._local_model = SentenceTransformer("intfloat/multilingual-e5-small")
        return self._local_model.encode(f"{prefix}: {text}", normalize_embeddings=True).tolist()
    
    def _classify_topic(self, chunk: Chunk) -> str:
        """
        Classify a chunk into a topic collection.
        
        Simple rule-based classification for now. Could be enhanced with ML.
        """
        text_lower = chunk.text.lower()
        
        # Count keyword matches per topic
        topic_scores = {}
        for topic, keywords in self.topic_rules.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                topic_scores[topic] = score
        
        # Use topic hints from chunking if available
        if chunk.topic_hints:
            for hint in chunk.topic_hints:
                hint_lower = hint.lower()
                for topic, keywords in self.topic_rules.items():
                    if any(hint_lower in keyword or keyword in hint_lower for keyword in keywords):
                        topic_scores[topic] = topic_scores.get(topic, 0) + 2
        
        # Return best match or default
        if topic_scores:
            return max(topic_scores.items(), key=lambda x: x[1])[0]
        else:
            return "memory_general"
    
    def create_collection(self, name: str, description: str = "") -> bool:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            description: Optional description of the collection's purpose
        
        Returns:
            True if created successfully, False if already exists
        """
        try:
            # Check if collection already exists
            existing_collections = [c.name for c in self.chroma_client.list_collections()]
            if name in existing_collections:
                return False
            
            # Create collection with metadata
            metadata = {"description": description} if description else None
            self.chroma_client.create_collection(
                name=name,
                metadata=metadata
            )
            return True
        except Exception as e:
            print(f"Error creating collection {name}: {e}")
            return False
    
    def ensure_default_collections(self):
        """Ensure all default collections exist."""
        for name, description in self.default_collections.items():
            self.create_collection(name, description)
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections with their metadata and stats."""
        collections = []
        for collection in self.chroma_client.list_collections():
            try:
                col_obj = self.chroma_client.get_collection(collection.name)
                collections.append({
                    "name": collection.name,
                    "metadata": collection.metadata or {},
                    "count": col_obj.count()
                })
            except Exception as e:
                print(f"Error accessing collection {collection.name}: {e}")
                collections.append({
                    "name": collection.name,
                    "metadata": collection.metadata or {},
                    "count": 0,
                    "error": str(e)
                })
        return collections
    
    def ingest_chunks(
        self,
        chunks: List[Chunk],
        source: str,
        auto_classify: bool = True,
        target_collection: Optional[str] = None
    ) -> Dict[str, int]:
        """
        Ingest chunks into appropriate collections.
        
        Args:
            chunks: List of chunks to ingest
            source: Source identifier (filename, URL, etc.)
            auto_classify: Whether to automatically classify chunks by topic
            target_collection: Force all chunks into this collection (overrides auto_classify)
        
        Returns:
            Dict with collection names and number of chunks added to each
        """
        # Ensure default collections exist
        self.ensure_default_collections()
        
        ingestion_stats = {}
        
        for chunk in chunks:
            # Determine target collection
            if target_collection:
                collection_name = target_collection
            elif auto_classify:
                collection_name = self._classify_topic(chunk)
            else:
                collection_name = "memory_general"
            
            # Get the collection
            try:
                collection = self.chroma_client.get_collection(collection_name)
            except Exception:
                # Create collection if it doesn't exist
                self.create_collection(collection_name)
                collection = self.chroma_client.get_collection(collection_name)
            
            # Generate embedding
            try:
                embedding = self._get_embedding(chunk.text[:1000], "passage")  # Limit to 1000 chars
            except Exception as e:
                print(f"Error generating embedding for chunk: {e}")
                continue
            
            # Prepare metadata
            metadata = {
                "source": source,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "speakers": ",".join(chunk.speakers) if chunk.speakers else "",
                "topic_hints": ",".join(chunk.topic_hints) if chunk.topic_hints else "",
                "char_count": len(chunk.text)
            }
            
            if chunk.timestamp:
                metadata["timestamp"] = chunk.timestamp.isoformat()
            
            # Add to collection
            try:
                chunk_id = str(uuid.uuid4())
                collection.add(
                    ids=[chunk_id],
                    documents=[chunk.text],
                    metadatas=[metadata],
                    embeddings=[embedding]
                )
                
                # Update stats
                ingestion_stats[collection_name] = ingestion_stats.get(collection_name, 0) + 1
                
            except Exception as e:
                print(f"Error adding chunk to {collection_name}: {e}")
                continue
        
        return ingestion_stats
    
    def search_collections(
        self,
        query: str,
        collections: Optional[List[str]] = None,
        n_results: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search across specified collections or all collections.
        
        Args:
            query: Search query
            collections: List of collection names to search (None = all)
            n_results: Number of results per collection
        
        Returns:
            Dict with collection names as keys and search results as values
        """
        if collections is None:
            collections = [c["name"] for c in self.list_collections()]
        
        query_embedding = self._get_embedding(query, "query")
        results = {}
        
        for collection_name in collections:
            try:
                collection = self.chroma_client.get_collection(collection_name)
                if collection.count() == 0:
                    continue
                
                search_results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["documents", "metadatas", "distances"]
                )
                
                if search_results["documents"]:
                    results[collection_name] = []
                    for i, doc in enumerate(search_results["documents"][0]):
                        results[collection_name].append({
                            "text": doc,
                            "metadata": search_results["metadatas"][0][i],
                            "distance": search_results["distances"][0][i]
                        })
                
            except Exception as e:
                print(f"Error searching collection {collection_name}: {e}")
                continue
        
        return results
    
    def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        try:
            self.chroma_client.delete_collection(name)
            return True
        except Exception as e:
            print(f"Error deleting collection {name}: {e}")
            return False
    
    def get_collection_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed statistics for a collection."""
        try:
            collection = self.chroma_client.get_collection(name)
            count = collection.count()
            
            if count == 0:
                return {"name": name, "count": 0, "sources": [], "speakers": []}
            
            # Get sample of documents to analyse
            sample_size = min(count, 100)
            results = collection.get(limit=sample_size, include=["metadatas"])
            
            # Analyse metadata
            sources = set()
            speakers = set()
            
            for metadata in results["metadatas"]:
                if metadata.get("source"):
                    sources.add(metadata["source"])
                if metadata.get("speakers"):
                    speakers.update(metadata["speakers"].split(","))
            
            return {
                "name": name,
                "count": count,
                "sources": list(sources),
                "speakers": [s for s in speakers if s.strip()],
            }
            
        except Exception as e:
            print(f"Error getting stats for collection {name}: {e}")
            return None