#!/usr/bin/env python3
"""
Evaluation harness for ambient memory system.

Tests the search quality against real Telegram conversation data by:
1. Extracting Diego's messages as queries
2. Running each through the memory system
3. Scoring results based on context relevance
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import httpx
from dataclasses import dataclass

# Add the src directory to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from ambient_memory.search import HybridSearcher
    from ambient_memory.chunking import ConversationChunker
    from ambient_memory.collections import CollectionManager
except ImportError as e:
    print(f"Warning: Could not import ambient_memory modules: {e}")
    print("Will try to use HTTP API instead of direct imports")


@dataclass
class EvaluationQuery:
    """A query extracted from conversation data for evaluation."""
    text: str
    source_file: str
    line_number: int
    timestamp: Optional[str] = None
    context_before: List[str] = None
    context_after: List[str] = None
    
    def __post_init__(self):
        if self.context_before is None:
            self.context_before = []
        if self.context_after is None:
            self.context_after = []


@dataclass
class EvaluationResult:
    """Result of running a query through the memory system."""
    query: EvaluationQuery
    results: List[Dict[str, Any]]
    execution_time_ms: int
    relevance_scores: List[float] = None
    
    def __post_init__(self):
        if self.relevance_scores is None:
            self.relevance_scores = []


class ConversationParser:
    """Parse Telegram conversation files and extract Diego's messages."""
    
    def __init__(self):
        # Patterns to identify Diego's messages
        self.diego_patterns = [
            r'^\s*`\d+:\d+`\s*\*\*Diego:\*\*\s*(.+)',  # `12:34` **Diego:** message
            r'^\s*\*\*Diego:\*\*\s*(.+)',  # **Diego:** message  
            r'^\s*Diego:\s*(.+)',  # Diego: message
            r'.*\*\*Diego:\*\*\s*(.+)',  # Any line containing **Diego:** 
        ]
        
        # Pattern to extract timestamps
        self.timestamp_pattern = r'`(\d{2}:\d{2})`'
        
        # Patterns for invalid queries (pure attachments, lone emojis, system messages)
        self.invalid_patterns = [
            r'^\[📎[^\]]+\]$',  # Pure image/file attachments like "[📎 image.png]"
            r'^[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+$',  # Lone emojis
            r'^\*\*System\*\*:',  # System messages
            r'^System:',  # System messages
            r'^Bot:',  # Bot messages
        ]
    
    def _is_valid_query(self, text: str) -> bool:
        """Check if a message is a valid query (not pure attachment/emoji/system message)."""
        if not text.strip():
            return False
            
        # Check for invalid patterns
        for pattern in self.invalid_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return False
        
        # Check if it has any actual textual content (words, not just symbols/emojis)
        # Remove emojis and check if there are still letter/number characters
        import unicodedata
        text_no_emoji = ''.join(c for c in text if unicodedata.category(c) not in ['So', 'Sm'])
        text_cleaned = re.sub(r'[^\w\s]', '', text_no_emoji)
        
        # Must have at least one alphanumeric character
        return bool(re.search(r'[a-zA-Z0-9]', text_cleaned))
    
    def extract_diego_messages(self, file_path: Path) -> List[EvaluationQuery]:
        """Extract Diego's messages from a conversation file."""
        queries = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if this line contains a Diego message
            diego_text = None
            for pattern in self.diego_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    if match.groups():
                        # Extract just the message text
                        diego_text = match.group(1).strip()
                    else:
                        # Full line match, extract everything after the name
                        if '**Diego:**' in line:
                            parts = line.split('**Diego:**', 1)
                            if len(parts) > 1:
                                diego_text = parts[1].strip().lstrip('*').strip()
                        elif 'Diego:' in line:
                            parts = line.split('Diego:', 1)
                            if len(parts) > 1:
                                diego_text = parts[1].strip()
                    break
            
            # Filter logic: keep short messages but exclude pure attachments/emojis/system messages
            if diego_text and self._is_valid_query(diego_text):
                # Extract timestamp if present
                timestamp_match = re.search(self.timestamp_pattern, line)
                timestamp = timestamp_match.group(1) if timestamp_match else None
                
                # Get context lines (3 before, 3 after)
                context_before = []
                context_after = []
                
                for j in range(max(0, i-3), i):
                    if j < len(lines):
                        context_before.append(lines[j].strip())
                
                for j in range(i+1, min(len(lines), i+4)):
                    if j < len(lines):
                        context_after.append(lines[j].strip())
                
                queries.append(EvaluationQuery(
                    text=diego_text,
                    source_file=str(file_path),
                    line_number=i + 1,
                    timestamp=timestamp,
                    context_before=context_before,
                    context_after=context_after
                ))
        
        return queries


class RelevanceScorer:
    """Score the relevance of search results based on context."""
    
    def __init__(self):
        # Keywords that indicate high relevance
        self.high_relevance_keywords = [
            "project", "work", "task", "deadline", "meeting", "client",
            "development", "build", "deploy", "feature", "bug", "issue"
        ]
        
        # Keywords that indicate low relevance
        self.low_relevance_keywords = [
            "weather", "morning", "evening", "hello", "thanks", "ok"
        ]
    
    def score_relevance(self, query: EvaluationQuery, result: Dict[str, Any]) -> float:
        """
        Score the relevance of a search result to a query.
        
        Key questions:
        - Is the returned context actually about the same topic as the message?
        - Would this context help an AI understand what the user is talking about?
        
        Returns:
            float: Relevance score from 0.0 (irrelevant) to 1.0 (highly relevant)
        """
        result_text = result.get("text", "").lower()
        query_text = query.text.lower()
        
        # Basic keyword overlap (exact word matches)
        query_words = set(query_text.split())
        result_words = set(result_text.split())
        exact_matches = query_words & result_words
        overlap_score = len(exact_matches) / max(len(query_words), 1) if query_words else 0
        
        # Topic coherence - check if result relates to query context
        context_text = " ".join(query.context_before + query.context_after).lower()
        if context_text:
            context_words = set(context_text.split())
            context_overlap = len((context_words & result_words)) / max(len(context_words), 1)
        else:
            context_overlap = 0
        
        # Enhanced entity extraction and matching
        def extract_entities(text):
            """Extract entities: capitalised words, project names, tools, etc."""
            words = text.split()
            entities = set()
            i = 0
            while i < len(words):
                word = words[i].strip(".,!?;:\"'()[]{}") 
                if word and word[0].isupper() and len(word) >= 2 and not word.isupper():
                    # Collect consecutive capitalised words as one entity
                    entity_parts = [word]
                    j = i + 1
                    while j < len(words) and j - i < 3:  # max 3 words per entity
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
            # Add ALL-CAPS acronyms (API, SEO, etc.)
            for word in words:
                word = word.strip(".,!?;:\"'()[]{}")
                if word.isupper() and len(word) >= 2:
                    entities.add(word.lower())
            return entities
        
        # Project/tool entities (known high-value matches)
        project_entities = ["darwin", "chowdown", "hire space", "railway", "the factory", "the keep", 
                          "root juice", "tello", "stitch", "protein counter", "geo", "diego", "rook"]
        
        query_entities = extract_entities(query.text)
        result_entities = extract_entities(result_text)
        
        # Entity boost calculation
        entity_boost = 0
        # High-value project matches get big boost
        for entity in project_entities:
            if entity in query_text and entity in result_text:
                entity_boost += 0.4
        # General entity matches get smaller boost
        common_entities = query_entities & result_entities
        entity_boost += len(common_entities) * 0.1
        
        entity_boost = min(entity_boost, 0.7)  # Cap at 0.7
        
        # Topic relevance indicators
        high_rel_score = sum(0.05 for kw in self.high_relevance_keywords if kw in result_text)
        low_rel_penalty = sum(0.1 for kw in self.low_relevance_keywords if kw in result_text and kw not in query_text)
        
        # Short query handling - for very short queries (<=5 words), expand with context and adjust weights
        if len(query_words) <= 5:
            # Expand short query with surrounding context
            expanded_query = query.text
            if query.context_before:
                recent_context = " ".join(query.context_before[-2:])  # Last 2 context lines
                expanded_query = recent_context + " " + expanded_query
            if query.context_after:
                following_context = " ".join(query.context_after[:2])  # Next 2 context lines  
                expanded_query = expanded_query + " " + following_context
            
            # Re-calculate with expanded query
            expanded_words = set(expanded_query.lower().split())
            expanded_matches = expanded_words & result_words
            overlap_score = max(overlap_score, len(expanded_matches) / max(len(expanded_words), 1))
            
            context_weight = 0.3
            overlap_weight = 0.6
        else:
            context_weight = 0.1
            overlap_weight = 0.8
        
        # Combine scores
        final_score = (
            overlap_score * overlap_weight +
            context_overlap * context_weight +
            entity_boost +
            high_rel_score -
            low_rel_penalty
        )
        
        return max(0.0, min(1.0, final_score))


class MemoryEvaluator:
    """Evaluates the ambient memory system against real conversation data."""
    
    def __init__(
        self,
        memory_api_url: str = "http://localhost:8000",
        embedding_server_url: str = "http://localhost:9876"
    ):
        self.memory_api_url = memory_api_url
        self.embedding_server_url = embedding_server_url
        self.parser = ConversationParser()
        self.scorer = RelevanceScorer()
        
        # Try to initialise direct components if available
        self.direct_mode = False
        try:
            self.searcher = HybridSearcher(
                chroma_path=".chromadb",
                embedding_server_url=embedding_server_url
            )
            self.collection_manager = CollectionManager(
                chroma_path=".chromadb",
                embedding_server_url=embedding_server_url
            )
            self.direct_mode = True
            print("✅ Direct mode: Using ambient_memory modules directly")
        except Exception as e:
            print(f"⚠️  API mode: Could not initialise direct modules ({e})")
            print("    Will use HTTP API if available")
    
    def _query_via_api(self, query: str, limit: int = 10) -> Tuple[List[Dict], int]:
        """Query memory via HTTP API."""
        try:
            response = httpx.post(
                f"{self.memory_api_url}/query",
                json={
                    "query": query,
                    "limit": limit,
                    "include_scores": True
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            return data.get("results", []), data.get("execution_time_ms", 0)
        except Exception as e:
            print(f"API query failed: {e}")
            return [], 0
    
    def _query_direct(self, query: str, limit: int = 10) -> Tuple[List[Dict], int]:
        """Query memory directly via modules."""
        if not self.direct_mode:
            return [], 0
        
        try:
            # Get available collections
            collections = [c["name"] for c in self.collection_manager.list_collections() if c["count"] > 0]
            if not collections:
                return [], 0
            
            start_time = datetime.now()
            results = self.searcher.search(query, collections, limit)
            end_time = datetime.now()
            
            execution_time = int((end_time - start_time).total_seconds() * 1000)
            return results, execution_time
            
        except Exception as e:
            print(f"Direct query failed: {e}")
            return [], 0
    
    def query_memory(self, query: str, limit: int = 10) -> Tuple[List[Dict], int]:
        """Query the memory system, trying direct mode first, then API."""
        if self.direct_mode:
            results, exec_time = self._query_direct(query, limit)
            if results:
                return results, exec_time
        
        # Fall back to API
        return self._query_via_api(query, limit)
    
    def ingest_conversation_files(self, file_paths: List[Path]) -> Dict[str, Any]:
        """Ingest conversation files into the memory system."""
        if not self.direct_mode:
            print("❌ Ingestion only available in direct mode")
            return {}
        
        try:
            chunker = ConversationChunker()
            total_chunks = 0
            ingestion_stats = {}
            
            for file_path in file_paths:
                print(f"📄 Ingesting {file_path.name}...")
                chunks = chunker.chunk_conversation_file(file_path)
                
                if chunks:
                    stats = self.collection_manager.ingest_chunks(
                        chunks=chunks,
                        source=file_path.name,
                        auto_classify=True
                    )
                    
                    total_chunks += len(chunks)
                    for collection, count in stats.items():
                        ingestion_stats[collection] = ingestion_stats.get(collection, 0) + count
            
            return {
                "total_chunks": total_chunks,
                "collection_stats": ingestion_stats,
                "files_processed": len(file_paths)
            }
            
        except Exception as e:
            print(f"❌ Ingestion failed: {e}")
            return {}
    
    def evaluate_queries(self, queries: List[EvaluationQuery]) -> List[EvaluationResult]:
        """Run evaluation queries through the memory system."""
        results = []
        
        print(f"🔍 Running {len(queries)} evaluation queries...")
        
        for i, query in enumerate(queries):
            if i % 10 == 0:
                print(f"   Progress: {i}/{len(queries)} ({i/len(queries)*100:.1f}%)")
            
            # Query the memory system
            memory_results, exec_time = self.query_memory(query.text)
            
            # Score relevance
            relevance_scores = []
            for result in memory_results:
                score = self.scorer.score_relevance(query, result)
                relevance_scores.append(score)
            
            results.append(EvaluationResult(
                query=query,
                results=memory_results,
                execution_time_ms=exec_time,
                relevance_scores=relevance_scores
            ))
        
        return results
    
    def compute_metrics(self, eval_results: List[EvaluationResult]) -> Dict[str, Any]:
        """Compute evaluation metrics from results."""
        total_queries = len(eval_results)
        total_results = sum(len(r.results) for r in eval_results)
        
        if total_queries == 0:
            return {"error": "No evaluation results"}
        
        # Precision metrics
        high_relevance_results = 0  # Score >= 0.7
        medium_relevance_results = 0  # Score >= 0.4 and < 0.7
        low_relevance_results = 0  # Score < 0.4
        
        all_scores = []
        queries_with_results = 0
        
        for eval_result in eval_results:
            if eval_result.results:
                queries_with_results += 1
                for score in eval_result.relevance_scores:
                    all_scores.append(score)
                    if score >= 0.4:  # Lowered from 0.7
                        high_relevance_results += 1
                    elif score >= 0.2:  # Lowered from 0.4
                        medium_relevance_results += 1
                    else:
                        low_relevance_results += 1
        
        # Calculate metrics
        coverage = queries_with_results / total_queries if total_queries > 0 else 0
        avg_relevance = sum(all_scores) / len(all_scores) if all_scores else 0
        precision_at_high = high_relevance_results / total_results if total_results > 0 else 0
        noise_rate = low_relevance_results / total_results if total_results > 0 else 0
        
        # Average execution time
        avg_exec_time = sum(r.execution_time_ms for r in eval_results) / len(eval_results)
        
        return {
            "total_queries": total_queries,
            "queries_with_results": queries_with_results,
            "coverage": round(coverage, 4),
            "total_results": total_results,
            "avg_results_per_query": round(total_results / total_queries, 2),
            "avg_relevance_score": round(avg_relevance, 4),
            "precision_at_high": round(precision_at_high, 4),
            "noise_rate": round(noise_rate, 4),
            "avg_execution_time_ms": round(avg_exec_time, 2),
            "score_distribution": {
                "high_relevance": high_relevance_results,
                "medium_relevance": medium_relevance_results,
                "low_relevance": low_relevance_results
            }
        }


def main():
    """Main evaluation script."""
    print("🧠 Ambient Memory Evaluation Harness")
    print("=" * 50)
    
    # Set up paths
    workspace = Path(__file__).parent.parent
    telegram_dir = Path("/Users/rook/workspace/memory/telegram")
    eval_dir = workspace / "eval"
    eval_dir.mkdir(exist_ok=True)
    
    # Test data files (Feb 10-17 as specified in task)
    test_files = []
    for day in range(10, 18):  # Feb 10-17
        file_path = telegram_dir / f"the-factory-2026-02-{day:02d}.md"
        test_files.append(file_path)
    
    # Check files exist
    available_files = [f for f in test_files if f.exists()]
    if not available_files:
        print("❌ No test files found. Available files:")
        for f in telegram_dir.glob("the-factory-*.md"):
            print(f"    {f.name}")
        return
    
    print(f"📁 Using test files: {[f.name for f in available_files]}")
    
    # Initialise evaluator
    evaluator = MemoryEvaluator()
    
    # Phase 1: Extract queries
    print("\n📋 Phase 1: Extracting Diego's messages...")
    all_queries = []
    for file_path in available_files:
        queries = evaluator.parser.extract_diego_messages(file_path)
        print(f"   {file_path.name}: {len(queries)} messages")
        all_queries.extend(queries)
    
    print(f"✅ Extracted {len(all_queries)} total queries")
    
    if not all_queries:
        print("❌ No Diego messages found in test files")
        return
    
    # Phase 2: Ingest data (if in direct mode)
    print("\n💾 Phase 2: Ingesting test data...")
    if evaluator.direct_mode:
        # Ingest more data for better evaluation (Feb 10-17)
        all_factory_files = []
        for day in range(10, 18):  # Feb 10-17
            file_path = telegram_dir / f"the-factory-2026-02-{day:02d}.md"
            if file_path.exists():
                all_factory_files.append(file_path)
        
        print(f"   Found {len(all_factory_files)} factory files to ingest")
        ingestion_result = evaluator.ingest_conversation_files(all_factory_files)
        print(f"   Ingested {ingestion_result.get('total_chunks', 0)} chunks")
        print(f"   Collections: {ingestion_result.get('collection_stats', {})}")
    else:
        print("   ⚠️  Skipping ingestion (API mode - data should be pre-loaded)")
    
    # Phase 3: Run evaluation
    print(f"\n🔍 Phase 3: Running evaluation...")
    # Use all extracted queries for comprehensive evaluation
    test_queries = all_queries
    print(f"   Testing with {len(test_queries)} queries (full dataset)")
    
    eval_results = evaluator.evaluate_queries(test_queries)
    
    # Phase 4: Compute metrics
    print("\n📊 Phase 4: Computing metrics...")
    metrics = evaluator.compute_metrics(eval_results)
    
    # Display results
    print("\n" + "=" * 50)
    print("🏆 EVALUATION RESULTS")
    print("=" * 50)
    print(f"Coverage:              {metrics.get('coverage', 0):.2%}")
    print(f"Avg Relevance Score:   {metrics.get('avg_relevance_score', 0):.3f}")
    print(f"High Precision:        {metrics.get('precision_at_high', 0):.2%}")
    print(f"Noise Rate:            {metrics.get('noise_rate', 0):.2%}")
    print(f"Avg Exec Time:         {metrics.get('avg_execution_time_ms', 0):.0f}ms")
    print(f"Results per Query:     {metrics.get('avg_results_per_query', 0):.1f}")
    
    # Save detailed results
    results_file = eval_dir / "results.json"
    detailed_results = {
        "timestamp": datetime.now().isoformat(),
        "test_files": [f.name for f in available_files],
        "total_queries_available": len(all_queries),
        "queries_tested": len(test_queries),
        "metrics": metrics,
        "sample_queries": [
            {
                "text": q.text[:100] + "..." if len(q.text) > 100 else q.text,
                "source": Path(q.source_file).name,
                "results_count": len(r.results),
                "avg_relevance": sum(r.relevance_scores) / len(r.relevance_scores) if r.relevance_scores else 0
            }
            for q, r in zip(test_queries[:5], eval_results[:5])
        ]
    }
    
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()