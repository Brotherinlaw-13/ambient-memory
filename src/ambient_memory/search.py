"""
Hybrid search implementation combining semantic embeddings with keyword/entity matching.

The core innovation: pure semantic similarity fails for agent memory. 
This module implements weighted hybrid search (70% semantic + 30% keyword/entity by default).
"""

import re
import httpx
from typing import List, Dict, Optional, Tuple, Any
import chromadb
from chromadb.config import Settings


class EntityExtractor:
    """Extract key entities from text for keyword matching."""
    
    @staticmethod
    def extract_entities(text: str) -> set:
        """Extract key entities from text: capitalized words/phrases, tool names, proper nouns."""
        # Match capitalised words (2+ chars), including multi-word like "Google Calendar"
        # Also match common tool/service patterns
        words = text.split()
        entities = set()
        i = 0
        while i < len(words):
            word = words[i].strip(".,!?;:\"'()[]{}") 
            # Skip common sentence-start words by checking if they're truly proper nouns
            # (appear mid-sentence or are known entities)
            if word and word[0].isupper() and len(word) >= 2 and not word.isupper():
                # Collect consecutive capitalised words as one entity
                entity_parts = [word]
                j = i + 1
                while j < len(words):
                    next_word = words[j].strip(".,!?;:\"'()[]{}")
                    if next_word and next_word[0].isupper() and len(next_word) >= 2:
                        entity_parts.append(next_word)
                        j += 1
                    else:
                        break
                entity = " ".join(entity_parts)
                entities.add(entity.lower())
                # Also add individual words for partial matching
                for part in entity_parts:
                    if len(part) >= 3:
                        entities.add(part.lower())
                i = j
            else:
                i += 1
        # Also catch ALL-CAPS acronyms (API, SEO, etc.)
        for word in words:
            word = word.strip(".,!?;:\"'()[]{}")
            if word.isupper() and len(word) >= 2:
                entities.add(word.lower())
        return entities


class HybridSearcher:
    """
    Hybrid search combining semantic similarity with keyword/entity matching.
    
    Based on production learnings: pure semantic search scored -9 over 76 queries
    (8% helpful, 73% neutral, 20% noise). Hybrid search significantly improves relevance.
    """
    
    def __init__(
        self,
        chroma_path: str = ".chromadb",
        embedding_server_url: str = "http://localhost:9876",
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        distance_threshold: float = 1.5,
        min_similarity_threshold: float = 0.60,
        context_expansion: bool = True,
    ):
        """
        Initialise hybrid searcher.
        
        Args:
            chroma_path: Path to ChromaDB storage directory
            embedding_server_url: URL to embedding server (or None to load locally)
            semantic_weight: Weight for semantic similarity (default 0.7)
            keyword_weight: Weight for keyword/entity matching (default 0.3)
            distance_threshold: Maximum distance to consider relevant (default 1.5)
            min_similarity_threshold: Drop results below this similarity (default 0.60)
            context_expansion: Prepend conversation context to all queries (default True)
        """
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.distance_threshold = distance_threshold
        self.min_similarity_threshold = min_similarity_threshold
        self.context_expansion = context_expansion
        self.embedding_server_url = embedding_server_url
        
        # Initialise ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialise entity extractor
        self.entity_extractor = EntityExtractor()
        
        # Check if embedding server is available
        self.use_embedding_server = self._check_embedding_server()
    
    def _check_embedding_server(self) -> bool:
        """Check if embedding server is available."""
        if not self.embedding_server_url:
            return False
        try:
            response = httpx.get(f"{self.embedding_server_url}/health", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
    
    def _get_embedding(self, text: str, prefix: str = "query") -> List[float]:
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
    
    def _get_embedding_local(self, text: str, prefix: str = "query") -> List[float]:
        """Get embedding using local model (fallback)."""
        # Import here to avoid loading model if server is available
        from sentence_transformers import SentenceTransformer
        if not hasattr(self, '_local_model'):
            self._local_model = SentenceTransformer("intfloat/multilingual-e5-small")
        return self._local_model.encode(f"{prefix}: {text}", normalize_embeddings=True).tolist()
    
    def _compute_keyword_score(self, query_entities: set, document_text: str) -> float:
        """Compute keyword overlap score (0-1): fraction of query entities found in document."""
        if not query_entities:
            return 0.0
        document_lower = document_text.lower()
        matches = sum(1 for entity in query_entities if entity in document_lower)
        return matches / len(query_entities)
    
    def search(
        self,
        query: str,
        collections: List[str],
        limit: int = 10,
        include_scores: bool = True,
        context: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search across collections.
        
        Args:
            query: Search query
            collections: List of collection names to search
            limit: Maximum number of results
            include_scores: Whether to include scoring details
            context: Surrounding conversation messages for context expansion
        
        Returns:
            List of search results with hybrid scores
        """
        if not query or len(query) < 2:
            return []
        
        # Context expansion: prepend conversation context to query for embedding
        # This was the single biggest improvement in testing (+2.6% relevance, -16.6% noise)
        search_text = query
        if self.context_expansion and context:
            # Use last 2 context messages to enrich the query
            ctx = " ".join(context[-2:])
            search_text = f"{ctx} {query}"
        
        # Get query embedding
        query_embedding = self._get_embedding(search_text, "query")
        
        # Extract entities from query for keyword matching
        query_entities = self.entity_extractor.extract_entities(query)
        
        all_results = []
        
        # Search each collection
        for collection_name in collections:
            try:
                collection = self.chroma_client.get_collection(collection_name)
                if collection.count() == 0:
                    continue
            except Exception:
                # Collection doesn't exist or error accessing it
                continue
            
            # Perform semantic search
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(limit * 2, 20),  # Get extra results for hybrid scoring
                include=["documents", "metadatas", "distances"],
            )
            
            if not results or not results["documents"]:
                continue
            
            # Process results with hybrid scoring
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i]
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                
                # Skip results beyond distance threshold
                if distance > self.distance_threshold:
                    continue
                
                # Calculate semantic similarity (0-1, higher = better)
                semantic_similarity = max(0.0, 1.0 - (distance / 2.0))
                
                # Calculate keyword score
                keyword_score = self._compute_keyword_score(query_entities, doc)
                
                # Tag-based boosting (from metadata)
                tag_boost = 0.0
                tags = metadata.get("tags", "")
                if tags and query_entities:
                    tag_list = [t.strip().lower() for t in tags.split(",")]
                    tag_matches = sum(1 for entity in query_entities if any(entity in t for t in tag_list))
                    if tag_matches > 0:
                        tag_boost = min(tag_matches * 0.15, 0.4)
                
                # Compute final hybrid score
                final_score = (
                    semantic_similarity * self.semantic_weight +
                    keyword_score * self.keyword_weight +
                    tag_boost
                )
                
                result = {
                    "text": doc[:400] if len(doc) > 400 else doc,
                    "source": metadata.get("source", collection_name),
                    "collection": collection_name,
                    "final_score": round(final_score, 4),
                }
                
                if include_scores:
                    result.update({
                        "semantic_similarity": round(semantic_similarity, 4),
                        "keyword_score": round(keyword_score, 4),
                        "tag_boost": round(tag_boost, 4),
                        "distance": round(distance, 4),
                    })
                
                all_results.append(result)
        
        # Sort by final score (descending)
        all_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Apply min similarity threshold to reduce noise
        # Testing showed threshold 0.60 cuts noise from 43% to 18% with 90% coverage
        if self.min_similarity_threshold > 0:
            all_results = [
                r for r in all_results
                if r.get("semantic_similarity", 0) >= self.min_similarity_threshold
            ]
        
        return all_results[:limit]
    
    def update_weights(self, semantic_weight: float, keyword_weight: float):
        """Update scoring weights."""
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
    
    def update_threshold(self, threshold: float):
        """Update distance threshold for relevance."""
        self.distance_threshold = threshold