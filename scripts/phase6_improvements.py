#!/usr/bin/env python3
"""
Phase 6 improvements evaluation.
Tests specific enhancements to reduce noise and improve relevance.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
sys.path.append('/Users/rook/workspace/ambient-memory/scripts')

from simple_phase3_eval import MessageExtractor, Phase3Scorer, Query


class Phase6Improvements:
    """Implements Phase 6 improvements over Phase 3 baseline."""
    
    def __init__(self, scorer: Phase3Scorer):
        self.scorer = scorer
    
    def should_skip_query(self, query: Query) -> bool:
        """
        D. Query-type awareness: Skip queries that don't need memory.
        Short acknowledgements with no context should return nothing.
        """
        text_lower = query.text.lower()
        
        # Short acknowledgements that typically don't need memory
        skip_patterns = [
            "dale", "ok", "si", "no", "yes", "good", "nice", "thanks", "gracias",
            "me gusta", "perfecto", "genial", "bueno", "vale", "claro",
            "👍", "👌", "😊", "😂", "🔥", "💯"
        ]
        
        # If the query is very short and matches skip patterns
        if len(query.text.split()) <= 2:
            if any(pattern in text_lower for pattern in skip_patterns):
                # But keep if there's meaningful context around it
                context = " ".join(query.context_before[-2:] + query.context_after[:2]).lower()
                if not any(proj in context for proj in self.scorer.project_entities):
                    return True
        
        return False
    
    def apply_min_threshold(self, results: List[Dict[str, Any]], threshold: float = 0.55) -> List[Dict[str, Any]]:
        """
        A. Min relevance threshold: Drop results below threshold.
        """
        return [r for r in results if r.get('score', 0) >= threshold]
    
    def apply_adaptive_count(self, results: List[Dict[str, Any]], gap_threshold: float = 0.2) -> List[Dict[str, Any]]:
        """
        B. Adaptive result count: If gap between result 1 and 2 is large, only return result 1.
        """
        if len(results) < 2:
            return results
        
        # Sort by score descending
        sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        
        # Check gap between top 2 results
        if len(sorted_results) >= 2:
            gap = sorted_results[0].get('score', 0) - sorted_results[1].get('score', 0)
            if gap > gap_threshold:
                return [sorted_results[0]]  # Return only the best result
        
        return sorted_results
    
    def apply_deduplication(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        C. Result deduplication: If 2 results are from same source, keep only higher-scored.
        """
        seen_sources = {}
        deduplicated = []
        
        # Sort by score descending first
        sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
        
        for result in sorted_results:
            source = result.get('source', 'unknown')
            if source not in seen_sources:
                seen_sources[source] = True
                deduplicated.append(result)
        
        return deduplicated


class Phase6Evaluator:
    """Enhanced evaluator with Phase 6 improvements."""
    
    def __init__(self):
        self.extractor = MessageExtractor()
        self.scorer = Phase3Scorer()
        self.improvements = Phase6Improvements(self.scorer)
    
    def generate_realistic_results(self, query: Query) -> List[Dict[str, Any]]:
        """Generate more realistic search results for testing."""
        results = []
        text_lower = query.text.lower()
        
        # High relevance: Project-related result
        if any(proj in text_lower for proj in self.scorer.project_entities):
            for proj in self.scorer.project_entities:
                if proj in text_lower:
                    results.append({
                        'text': f"Discussion about {proj} project development, features, and implementation details.",
                        'source': f"{proj}_conversations",
                        'score': 0  # Will be calculated by scorer
                    })
                    break
        
        # Medium relevance: Context-related
        context = " ".join(query.context_before + query.context_after).lower()
        if any(proj in context for proj in self.scorer.project_entities):
            results.append({
                'text': f"Related conversation context about work and projects mentioned nearby.",
                'source': "general_work",
                'score': 0
            })
        
        # Low relevance: Generic results (simulating ChromaDB noise)
        generic_results = [
            {
                'text': f"General discussion about daily tasks and scheduling.",
                'source': "daily_chats", 
                'score': 0
            },
            {
                'text': f"Random conversation about various topics and casual chat.",
                'source': "casual_conversation",
                'score': 0
            },
            {
                'text': f"Technical discussion about software development and coding.",
                'source': "tech_talk",
                'score': 0
            }
        ]
        
        # Add 1-2 generic results to simulate noise
        results.extend(generic_results[:2])
        
        # Calculate actual scores
        for result in results:
            result['score'] = self.scorer.score_relevance(query, result['text'])
        
        return results
    
    def evaluate_improvement(self, queries: List[Query], improvement_name: str, apply_improvement) -> Dict[str, Any]:
        """Evaluate a specific improvement."""
        print(f"\n🔍 Testing {improvement_name}...")
        
        all_scores = []
        high_rel = 0
        med_rel = 0  
        low_rel = 0
        total_results = 0
        queries_with_results = 0
        skipped_queries = 0
        
        for i, query in enumerate(queries):
            if i % 50 == 0 and i > 0:
                print(f"  Progress: {i}/{len(queries)}")
            
            # Check if query should be skipped (improvement D)
            if improvement_name == "Query-type awareness" and self.improvements.should_skip_query(query):
                skipped_queries += 1
                continue
            
            # Generate results
            results = self.generate_realistic_results(query)
            
            # Apply improvement
            if improvement_name != "Query-type awareness":  # Already applied above
                results = apply_improvement(results)
            
            if results:
                queries_with_results += 1
                total_results += len(results)
                
                for result in results:
                    score = result['score']
                    all_scores.append(score)
                    
                    if score >= 0.4:
                        high_rel += 1
                    elif score >= 0.2:
                        med_rel += 1
                    else:
                        low_rel += 1
        
        # Calculate metrics
        coverage = queries_with_results / len(queries) if len(queries) > 0 else 0
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        precision = high_rel / total_results if total_results > 0 else 0
        noise_rate = low_rel / total_results if total_results > 0 else 0
        
        return {
            "improvement": improvement_name,
            "total_queries": len(queries),
            "skipped_queries": skipped_queries,
            "queries_with_results": queries_with_results,
            "coverage": coverage,
            "total_results": total_results,
            "avg_relevance_score": avg_score,
            "precision_at_high": precision,
            "noise_rate": noise_rate,
            "score_distribution": {
                "high_relevance": high_rel,
                "medium_relevance": med_rel,
                "low_relevance": low_rel
            }
        }


def main():
    print("🧠 Phase 6 Improvements Evaluation")
    print("=" * 40)
    
    # Extract queries
    telegram_dir = Path("/Users/rook/workspace/memory/telegram")
    evaluator = Phase6Evaluator()
    
    all_queries = []
    for day in range(10, 18):
        file_path = telegram_dir / f"the-factory-2026-02-{day:02d}.md"
        if file_path.exists():
            queries = evaluator.extractor.extract_from_file(file_path)
            all_queries.extend(queries)
    
    print(f"✅ Extracted {len(all_queries)} queries")
    
    # Test baseline (no improvements)
    print(f"\n📊 Baseline (Phase 3 logic)...")
    baseline_results = evaluator.evaluate_improvement(
        all_queries, 
        "Baseline", 
        lambda x: x  # No changes
    )
    
    # Test each improvement individually
    improvements = [
        ("Min threshold (0.50)", lambda results: evaluator.improvements.apply_min_threshold(results, 0.50)),
        ("Min threshold (0.55)", lambda results: evaluator.improvements.apply_min_threshold(results, 0.55)),
        ("Min threshold (0.60)", lambda results: evaluator.improvements.apply_min_threshold(results, 0.60)),
        ("Adaptive result count", evaluator.improvements.apply_adaptive_count),
        ("Result deduplication", evaluator.improvements.apply_deduplication),
        ("Query-type awareness", lambda x: x),  # Special case, handled in evaluate_improvement
    ]
    
    all_results = {"baseline": baseline_results}
    
    for name, improvement_func in improvements:
        result = evaluator.evaluate_improvement(all_queries, name, improvement_func)
        all_results[name.lower().replace(" ", "_").replace("(", "").replace(")", "")] = result
    
    # Display comparison
    print("\n" + "=" * 80)
    print("📊 PHASE 6 IMPROVEMENTS COMPARISON")
    print("=" * 80)
    print(f"{'Improvement':<25} {'Coverage':<10} {'Precision':<10} {'Noise':<10} {'Avg Score':<10}")
    print("-" * 80)
    
    for key, result in all_results.items():
        print(f"{result['improvement']:<25} "
              f"{result['coverage']:<9.1%} "
              f"{result['precision_at_high']:<9.1%} "
              f"{result['noise_rate']:<9.1%} "
              f"{result['avg_relevance_score']:<9.3f}")
    
    # Find best improvements
    print(f"\n🏆 Best Improvements:")
    
    # Best for reducing noise
    best_noise = min(all_results.values(), key=lambda x: x['noise_rate'])
    print(f"  Lowest noise: {best_noise['improvement']} ({best_noise['noise_rate']:.1%})")
    
    # Best for precision
    best_precision = max(all_results.values(), key=lambda x: x['precision_at_high'])
    print(f"  Highest precision: {best_precision['improvement']} ({best_precision['precision_at_high']:.1%})")
    
    # Best balanced (high precision, low noise)
    balanced_scores = {name: r['precision_at_high'] - r['noise_rate'] for name, r in all_results.items()}
    best_balanced_key = max(balanced_scores, key=balanced_scores.get)
    best_balanced = all_results[best_balanced_key]
    print(f"  Most balanced: {best_balanced['improvement']} (score: {balanced_scores[best_balanced_key]:.1%})")
    
    # Save results
    eval_dir = Path("eval")
    eval_dir.mkdir(exist_ok=True)
    
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "phase6_improvements",
        "total_queries": len(all_queries),
        "improvements_tested": list(all_results.keys()),
        "results": all_results,
        "best_improvements": {
            "lowest_noise": best_noise['improvement'],
            "highest_precision": best_precision['improvement'], 
            "most_balanced": best_balanced['improvement']
        }
    }
    
    results_file = eval_dir / "phase6-improvements.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("\n✅ Phase 6 improvements evaluation complete!")
    
    return final_results


if __name__ == "__main__":
    main()