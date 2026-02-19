#!/usr/bin/env python3
"""
Weight tuning script to test different keyword/context weight combinations.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the full evaluation classes  
from evaluate import MemoryEvaluator, ConversationParser, RelevanceScorer


class WeightTuningScorer(RelevanceScorer):
    """Extended scorer to test different weight combinations."""
    
    def __init__(self, overlap_weight=0.6, context_weight=0.2, entity_weight=1.0):
        super().__init__()
        self.overlap_weight = overlap_weight
        self.context_weight = context_weight  
        self.entity_weight = entity_weight
    
    def score_relevance(self, query, result):
        """Score with configurable weights."""
        result_text = result.get("text", "").lower()
        query_text = query.text.lower()
        
        # Basic keyword overlap (exact word matches)
        query_words = set(query_text.split())
        result_words = set(result_text.split())
        exact_matches = query_words & result_words
        overlap_score = len(exact_matches) / max(len(query_words), 1) if query_words else 0
        
        # Context relevance
        context_text = " ".join(query.context_before + query.context_after).lower()
        if context_text:
            context_words = set(context_text.split())
            context_overlap = len((context_words & result_words)) / max(len(context_words), 1)
        else:
            context_overlap = 0
        
        # Enhanced entity extraction and matching
        def extract_entities(text):
            words = text.split()
            entities = set()
            i = 0
            while i < len(words):
                word = words[i].strip(".,!?;:\"'()[]{}") 
                if word and word[0].isupper() and len(word) >= 2 and not word.isupper():
                    entity_parts = [word]
                    j = i + 1
                    while j < len(words) and j - i < 3:
                        next_word = words[j].strip(".,!?;:\"'()[]{}")
                        if next_word and next_word[0].isupper() and len(next_word) >= 2:
                            entity_parts.append(next_word)
                            j += 1
                        else:
                            break
                    entity = " ".join(entity_parts)
                    entities.add(entity.lower())
                    for part in entity_parts:
                        if len(part) >= 3:
                            entities.add(part.lower())
                    i = j
                else:
                    i += 1
            for word in words:
                word = word.strip(".,!?;:\"'()[]{}")
                if word.isupper() and len(word) >= 2:
                    entities.add(word.lower())
            return entities
        
        project_entities = ["darwin", "chowdown", "hire space", "railway", "the factory", "the keep",
                          "root juice", "tello", "stitch", "protein counter", "geo", "diego", "rook"]
        
        query_entities = extract_entities(query.text)
        result_entities = extract_entities(result_text)
        
        entity_boost = 0
        for entity in project_entities:
            if entity in query_text and entity in result_text:
                entity_boost += 0.4
        common_entities = query_entities & result_entities
        entity_boost += len(common_entities) * 0.1
        entity_boost = min(entity_boost, 0.7)
        
        # Topic relevance
        high_rel_score = sum(0.05 for kw in self.high_relevance_keywords if kw in result_text)
        low_rel_penalty = sum(0.1 for kw in self.low_relevance_keywords if kw in result_text and kw not in query_text)
        
        # Short query handling
        if len(query_words) <= 3:
            context_weight = self.context_weight * 2
            overlap_weight = self.overlap_weight * 0.8
        else:
            context_weight = self.context_weight
            overlap_weight = self.overlap_weight
        
        # Combine scores with configurable weights
        final_score = (
            overlap_score * overlap_weight +
            context_overlap * context_weight +
            entity_boost * self.entity_weight +
            high_rel_score -
            low_rel_penalty
        )
        
        return max(0.0, min(1.0, final_score))


def test_weight_combination(overlap_w, context_w, entity_w, queries, evaluator):
    """Test a specific weight combination."""
    # Replace the scorer in evaluator
    evaluator.scorer = WeightTuningScorer(overlap_w, context_w, entity_w)
    
    # Run evaluation
    eval_results = evaluator.evaluate_queries(queries)
    metrics = evaluator.compute_metrics(eval_results)
    
    return metrics.get('avg_relevance_score', 0), metrics.get('precision_at_high', 0), metrics.get('noise_rate', 0)


def main():
    print("🔧 Weight Tuning Experiment")
    print("=" * 40)
    
    # Setup
    telegram_dir = Path("/Users/rook/workspace/memory/telegram")
    test_files = [
        telegram_dir / "the-factory-2026-02-16.md",
        telegram_dir / "the-factory-2026-02-17.md"
    ]
    
    evaluator = MemoryEvaluator()
    
    # Get test queries (subset for speed)
    all_queries = []
    for file_path in test_files:
        if file_path.exists():
            queries = evaluator.parser.extract_diego_messages(file_path)
            all_queries.extend(queries)
    test_queries = all_queries[:20]  # Even smaller subset for speed
    
    print(f"Testing {len(test_queries)} queries with different weight combinations:")
    
    # Test different weight combinations
    combinations = [
        (0.6, 0.2, 1.0),  # Current (overlap, context, entity)
        (0.5, 0.3, 1.2),  # More context, more entity
        (0.8, 0.1, 0.8),  # More overlap, less others
        (0.4, 0.4, 1.5),  # Balanced overlap/context, high entity
        (0.7, 0.2, 0.5),  # High overlap, low entity
    ]
    
    results = []
    for i, (ov, ctx, ent) in enumerate(combinations):
        print(f"\n{i+1}/5 Testing weights: overlap={ov}, context={ctx}, entity={ent}")
        
        avg_rel, precision, noise = test_weight_combination(ov, ctx, ent, test_queries, evaluator)
        results.append((ov, ctx, ent, avg_rel, precision, noise))
        
        print(f"   → Avg relevance: {avg_rel:.3f}, High precision: {precision:.1%}, Noise: {noise:.1%}")
    
    # Find best combination
    best = max(results, key=lambda x: x[3])  # Best by avg relevance
    print(f"\n🏆 Best combination:")
    print(f"   Weights: overlap={best[0]}, context={best[1]}, entity={best[2]}")
    print(f"   Avg relevance: {best[3]:.3f}")
    print(f"   High precision: {best[4]:.1%}")
    print(f"   Noise rate: {best[5]:.1%}")
    
    return best


if __name__ == "__main__":
    main()