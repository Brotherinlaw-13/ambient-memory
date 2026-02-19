#!/usr/bin/env python3
"""
Phase 6 combined improvements - finding the optimal balance.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
sys.path.append('/Users/rook/workspace/ambient-memory/scripts')

from phase6_improvements import Phase6Evaluator, Phase6Improvements
from simple_phase3_eval import Query


class CombinedStrategy:
    """Combined improvement strategies with different threshold approaches."""
    
    def __init__(self, improvements: Phase6Improvements):
        self.improvements = improvements
    
    def balanced_approach(self, query: Query, results):
        """
        Balanced approach: lower threshold but with other filters.
        """
        # Skip acknowledgement-type queries
        if self.improvements.should_skip_query(query):
            return []
        
        # Apply deduplication first
        results = self.improvements.apply_deduplication(results)
        
        # Use adaptive count
        results = self.improvements.apply_adaptive_count(results)
        
        # Use lower threshold (0.30) for better coverage
        results = self.improvements.apply_min_threshold(results, threshold=0.30)
        
        return results
    
    def conservative_approach(self, query: Query, results):
        """
        Conservative: higher threshold but return at least one result if available.
        """
        # Skip acknowledgement-type queries
        if self.improvements.should_skip_query(query):
            return []
        
        # Try high threshold first
        filtered = self.improvements.apply_min_threshold(results, threshold=0.55)
        
        # If nothing passes, try medium threshold
        if not filtered:
            filtered = self.improvements.apply_min_threshold(results, threshold=0.35)
        
        # If still nothing, return best result if it's above minimum quality
        if not filtered and results:
            best_result = max(results, key=lambda x: x.get('score', 0))
            if best_result.get('score', 0) >= 0.20:
                filtered = [best_result]
        
        # Apply other improvements
        filtered = self.improvements.apply_deduplication(filtered)
        filtered = self.improvements.apply_adaptive_count(filtered)
        
        return filtered
    
    def smart_threshold(self, query: Query, results):
        """
        Smart threshold: adjust based on query characteristics.
        """
        # Skip acknowledgement-type queries
        if self.improvements.should_skip_query(query):
            return []
        
        text_lower = query.text.lower()
        
        # Higher threshold for project-specific queries (should have good matches)
        if any(proj in text_lower for proj in self.improvements.scorer.project_entities):
            threshold = 0.50
        # Medium threshold for longer queries
        elif len(query.text.split()) >= 8:
            threshold = 0.35
        # Lower threshold for short/general queries
        else:
            threshold = 0.25
        
        # Apply all improvements
        filtered = self.improvements.apply_deduplication(results)
        filtered = self.improvements.apply_adaptive_count(filtered)
        filtered = self.improvements.apply_min_threshold(filtered, threshold=threshold)
        
        return filtered


def main():
    print("🧠 Phase 6 Combined Strategies")
    print("=" * 35)
    
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
    
    # Test combined strategies
    combined = CombinedStrategy(evaluator.improvements)
    
    strategies = [
        ("Baseline", lambda q, r: r),
        ("Balanced (0.30 + filters)", combined.balanced_approach),
        ("Conservative (fallback)", combined.conservative_approach), 
        ("Smart threshold", combined.smart_threshold),
    ]
    
    all_results = {}
    
    for strategy_name, strategy_func in strategies:
        print(f"\n🔍 Testing {strategy_name}...")
        
        all_scores = []
        high_rel = 0
        med_rel = 0
        low_rel = 0
        total_results = 0
        queries_with_results = 0
        skipped_queries = 0
        
        for i, query in enumerate(all_queries):
            if i % 50 == 0 and i > 0:
                print(f"  Progress: {i}/{len(all_queries)}")
            
            # Generate realistic results
            results = evaluator.generate_realistic_results(query)
            
            # Apply strategy
            filtered_results = strategy_func(query, results)
            
            if strategy_name != "Baseline" and len(filtered_results) == 0 and len(results) > 0:
                # Count skipped queries (had results but strategy filtered them all)
                skipped_queries += 1
            
            if filtered_results:
                queries_with_results += 1
                total_results += len(filtered_results)
                
                for result in filtered_results:
                    score = result['score']
                    all_scores.append(score)
                    
                    if score >= 0.4:
                        high_rel += 1
                    elif score >= 0.2:
                        med_rel += 1
                    else:
                        low_rel += 1
        
        # Calculate metrics
        coverage = queries_with_results / len(all_queries)
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        precision = high_rel / total_results if total_results > 0 else 0
        noise_rate = low_rel / total_results if total_results > 0 else 0
        
        result = {
            "strategy": strategy_name,
            "total_queries": len(all_queries),
            "skipped_queries": skipped_queries,
            "queries_with_results": queries_with_results,
            "coverage": coverage,
            "total_results": total_results,
            "avg_relevance_score": avg_score,
            "precision_at_high": precision,
            "noise_rate": noise_rate,
            "results_per_query": total_results / len(all_queries) if len(all_queries) > 0 else 0,
            "score_distribution": {
                "high_relevance": high_rel,
                "medium_relevance": med_rel,
                "low_relevance": low_rel
            }
        }
        
        all_results[strategy_name.lower().replace(" ", "_").replace("(", "").replace(")", "")] = result
    
    # Display comparison
    print("\n" + "=" * 90)
    print("📊 COMBINED STRATEGIES COMPARISON") 
    print("=" * 90)
    print(f"{'Strategy':<25} {'Coverage':<10} {'Precision':<10} {'Noise':<10} {'Avg Score':<10} {'Res/Query':<10}")
    print("-" * 90)
    
    for result in all_results.values():
        print(f"{result['strategy']:<25} "
              f"{result['coverage']:<9.1%} "
              f"{result['precision_at_high']:<9.1%} "
              f"{result['noise_rate']:<9.1%} "
              f"{result['avg_relevance_score']:<9.3f} "
              f"{result['results_per_query']:<9.1f}")
    
    # Calculate quality scores (balance of precision and coverage)
    print(f"\n🏆 Strategy Quality Scores:")
    print(f"   (Formula: precision * coverage - noise_rate)")
    print("-" * 50)
    
    quality_scores = {}
    for name, result in all_results.items():
        quality_score = (result['precision_at_high'] * result['coverage']) - result['noise_rate']
        quality_scores[name] = quality_score
        print(f"   {result['strategy']:<25}: {quality_score:>6.3f}")
    
    best_strategy_key = max(quality_scores, key=quality_scores.get)
    best_strategy = all_results[best_strategy_key]
    
    print(f"\n🥇 Best Overall Strategy: {best_strategy['strategy']}")
    print(f"   Quality Score: {quality_scores[best_strategy_key]:.3f}")
    print(f"   Coverage: {best_strategy['coverage']:.1%}")
    print(f"   Precision: {best_strategy['precision_at_high']:.1%}")
    print(f"   Noise: {best_strategy['noise_rate']:.1%}")
    
    # Save results
    eval_dir = Path("eval")
    eval_dir.mkdir(exist_ok=True)
    
    final_results = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "phase6_combined_strategies",
        "total_queries": len(all_queries),
        "strategies": all_results,
        "quality_scores": quality_scores,
        "best_strategy": {
            "name": best_strategy['strategy'],
            "quality_score": quality_scores[best_strategy_key],
            "metrics": best_strategy
        }
    }
    
    results_file = eval_dir / "phase6-combined.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("\n✅ Combined strategies evaluation complete!")
    
    return final_results


if __name__ == "__main__":
    main()