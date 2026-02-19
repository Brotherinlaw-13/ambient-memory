"""
Improved hybrid search implementation combining Phase 3's working logic 
with Phase 5's structural improvements and Phase 6 enhancements.

This preserves the Phase 3 search algorithm that achieved 41.7% relevance
while adding clean configuration, utility functions, and new features.
"""

import re
import math
import httpx
from typing import List, Dict, Optional, Any, Set
from datetime import datetime
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings


@dataclass
class SearchConfig:
    """Configuration for HybridSearcher with Phase 3 defaults."""
    # Core Phase 3 weights (keep exactly the same)
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    distance_threshold: float = 1.5
    
    # Phase 6 improvements
    min_relevance_threshold: float = 0.0  # Disabled by default to preserve coverage
    enable_query_filtering: bool = True   # Skip acknowledgement queries
    enable_deduplication: bool = True     # Avoid duplicate sources
    adaptive_result_gap: float = 0.0      # Disabled by default, 0.2 to enable
    
    # Search limits
    max_results: int = 10
    
    # Context and temporal features (from Phase 5, optional)
    enable_temporal_boost: bool = False
    temporal_boost_weight: float = 0.1
    temporal_half_life_days: float = 3.0
    context_window: int = 3


def extract_entities(text: str) -> Set[str]:
    """
    Extract key entities from text (Phase 3 logic exactly preserved).
    Returns capitalised words/phrases, tool names, proper nouns.
    """
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


def compute_keyword_score(query_entities: Set[str], document_text: str) -> float:
    """
    Compute keyword overlap score (Phase 3 logic exactly preserved).
    Returns fraction of query entities found in document (0-1).
    """
    if not query_entities:
        return 0.0
    document_lower = document_text.lower()
    matches = sum(1 for entity in query_entities if entity in document_lower)
    return matches / len(query_entities)


def should_skip_query(query: str) -> bool:
    """
    Phase 6 improvement: Skip queries that don't need memory.
    Short acknowledgements with no meaningful context.
    """
    text_lower = query.lower()
    
    # Short acknowledgements that typically don't need memory
    skip_patterns = [
        "dale", "ok", "si", "no", "yes", "good", "nice", "thanks", "gracias",
        "me gusta", "perfecto", "genial", "bueno", "vale", "claro",
        "👍", "👌", "😊", "😂", "🔥", "💯"
    ]
    
    # If query is very short and matches skip patterns
    if len(query.split()) <= 2:
        return any(pattern in text_lower for pattern in skip_patterns)
    
    return False


def extract_date_from_source(source: str) -> Optional[datetime]:
    """Extract date from source filename like 'the-factory-2026-02-15.md'."""
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', source)
    if match:
        try:
            year, month, day = map(int, match.groups())
            return datetime(year, month, day)
        except ValueError:
            pass
    return None


def compute_temporal_boost(query_time: datetime, result_source: str, 
                         half_life_days: float = 3.0, weight: float = 0.1) -> float:
    """Compute temporal proximity boost using exponential decay."""
    result_date = extract_date_from_source(result_source)
    if not result_date:
        return 0.0
    
    # Calculate days difference
    days_diff = abs((query_time - result_date).days)
    
    # Exponential decay: boost = exp(-ln(2) * days_diff / half_life)
    boost = math.exp(-math.log(2) * days_diff / half_life_days)
    
    return boost * weight


class ImprovedHybridSearcher:
    """
    Hybrid searcher combining Phase 3's proven algorithm with structural improvements.
    
    Preserves the exact search logic that achieved 41.7% relevance and 25.9% noise,
    while adding clean configuration, utility functions, and Phase 6 enhancements.
    """
    
    def __init__(
        self,
        chroma_path: str = ".chromadb",
        embedding_server_url: str = "http://localhost:9876",
        config: Optional[SearchConfig] = None
    ):
        """Initialise improved hybrid searcher."""
        self.config = config or SearchConfig()
        self.embedding_server_url = embedding_server_url
        
        # Initialise ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=chroma_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
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
    
    def _apply_phase6_improvements(self, query: str, results: List[Dict[str, Any]], 
                                  query_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Apply Phase 6 improvements to filter and enhance results."""
        
        # Phase 6A: Query-type awareness - skip if enabled
        if self.config.enable_query_filtering and should_skip_query(query):
            return []
        
        if not results:
            return results
        
        # Phase 6C: Result deduplication
        if self.config.enable_deduplication:
            seen_sources = {}
            deduplicated = []
            # Sort by score first to keep best from each source
            sorted_results = sorted(results, key=lambda x: x.get("final_score", 0), reverse=True)
            
            for result in sorted_results:
                source = result.get("source", "unknown")
                if source not in seen_sources:
                    seen_sources[source] = True
                    deduplicated.append(result)
            
            results = deduplicated
        
        # Phase 6B: Adaptive result count
        if self.config.adaptive_result_gap > 0 and len(results) >= 2:
            sorted_results = sorted(results, key=lambda x: x.get("final_score", 0), reverse=True)
            gap = sorted_results[0].get("final_score", 0) - sorted_results[1].get("final_score", 0)
            if gap > self.config.adaptive_result_gap:
                results = [sorted_results[0]]  # Return only the best result
            else:
                results = sorted_results
        
        # Phase 6A: Min relevance threshold
        if self.config.min_relevance_threshold > 0:
            results = [r for r in results if r.get("final_score", 0) >= self.config.min_relevance_threshold]
        
        return results
    
    def search(
        self,
        query: str,
        collections: List[str],
        limit: Optional[int] = None,
        include_scores: bool = True,
        query_time: Optional[datetime] = None,
        context: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search with Phase 3 algorithm + Phase 6 improvements.
        
        Args:
            query: Search query
            collections: List of collection names to search
            limit: Maximum number of results (uses config default if None)
            include_scores: Whether to include scoring details
            query_time: Query timestamp for temporal boosting
            context: Surrounding messages for context expansion
        
        Returns:
            List of search results with hybrid scores
        """
        if not query or len(query) < 2:
            return []
        
        if limit is None:
            limit = self.config.max_results
        
        # Get query embedding
        query_embedding = self._get_embedding(query, "query")
        
        # Extract entities from query for keyword matching (Phase 3 logic)
        query_entities = extract_entities(query)
        
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
            
            # Process results with Phase 3 hybrid scoring (exactly preserved)
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i]
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                
                # Skip results beyond distance threshold (Phase 3)
                if distance > self.config.distance_threshold:
                    continue
                
                # Calculate semantic similarity (Phase 3 logic)
                semantic_similarity = max(0.0, 1.0 - (distance / 2.0))
                
                # Calculate keyword score (Phase 3 logic)
                keyword_score = compute_keyword_score(query_entities, doc)
                
                # Tag-based boosting (Phase 3 logic)
                tag_boost = 0.0
                tags = metadata.get("tags", "")
                if tags and query_entities:
                    tag_list = [t.strip().lower() for t in tags.split(",")]
                    tag_matches = sum(1 for entity in query_entities if any(entity in t for t in tag_list))
                    if tag_matches > 0:
                        tag_boost = min(tag_matches * 0.15, 0.4)
                
                # Optional temporal boost (from Phase 5)
                temporal_boost = 0.0
                if self.config.enable_temporal_boost and query_time:
                    temporal_boost = compute_temporal_boost(
                        query_time, 
                        metadata.get("source", ""),
                        self.config.temporal_half_life_days,
                        self.config.temporal_boost_weight
                    )
                
                # Compute final hybrid score (Phase 3 logic + optional temporal)
                final_score = (
                    semantic_similarity * self.config.semantic_weight +
                    keyword_score * self.config.keyword_weight +
                    tag_boost +
                    temporal_boost
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
                        "temporal_boost": round(temporal_boost, 4),
                        "distance": round(distance, 4),
                    })
                
                all_results.append(result)
        
        # Sort by final score (descending)
        all_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Apply Phase 6 improvements
        all_results = self._apply_phase6_improvements(query, all_results, query_time)
        
        # Return top results
        return all_results[:limit]
    
    def update_config(self, config: SearchConfig):
        """Update searcher configuration."""
        self.config = config


# Maintain compatibility with Phase 3 class name
HybridSearcher = ImprovedHybridSearcher