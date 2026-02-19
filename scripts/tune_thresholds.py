#!/usr/bin/env python3
"""
Threshold tuning script to test different relevance score thresholds.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import evaluation classes
from evaluate import MemoryEvaluator


def compute_metrics_with_thresholds(eval_results, high_threshold, medium_threshold):
    """Compute metrics with custom thresholds."""
    total_queries = len(eval_results)
    total_results = sum(len(r.results) for r in eval_results)
    
    if total_queries == 0:
        return {"error": "No evaluation results"}
    
    # Count by thresholds
    high_relevance_results = 0  
    medium_relevance_results = 0
    low_relevance_results = 0
    
    all_scores = []
    queries_with_results = 0
    
    for eval_result in eval_results:
        if eval_result.results:
            queries_with_results += 1
            for score in eval_result.relevance_scores:
                all_scores.append(score)
                if score >= high_threshold:
                    high_relevance_results += 1
                elif score >= medium_threshold:
                    medium_relevance_results += 1
                else:
                    low_relevance_results += 1
    
    # Calculate metrics
    coverage = queries_with_results / total_queries if total_queries > 0 else 0
    avg_relevance = sum(all_scores) / len(all_scores) if all_scores else 0
    precision_at_high = high_relevance_results / total_results if total_results > 0 else 0
    noise_rate = low_relevance_results / total_results if total_results > 0 else 0
    
    return {
        "avg_relevance_score": avg_relevance,
        "precision_at_high": precision_at_high,
        "noise_rate": noise_rate,
        "score_distribution": {
            "high_relevance": high_relevance_results,
            "medium_relevance": medium_relevance_results,
            "low_relevance": low_relevance_results
        }
    }


def main():
    print("📊 Threshold Tuning Experiment")
    print("=" * 40)
    
    # Setup
    telegram_dir = Path("/Users/rook/workspace/memory/telegram")
    test_files = [
        telegram_dir / "the-factory-2026-02-16.md",
        telegram_dir / "the-factory-2026-02-17.md"
    ]
    
    evaluator = MemoryEvaluator()
    
    # Get test queries 
    all_queries = []
    for file_path in test_files:
        if file_path.exists():
            queries = evaluator.parser.extract_diego_messages(file_path)
            all_queries.extend(queries)
    test_queries = all_queries[:25]  # Reasonable subset
    
    print(f"Testing {len(test_queries)} queries with different thresholds:")
    
    # Run evaluation once with current settings
    print("Running evaluation...")
    eval_results = evaluator.evaluate_queries(test_queries)
    
    # Test different threshold combinations
    threshold_combinations = [
        (0.7, 0.4),   # Current (high, medium)
        (0.6, 0.3),   # Lower thresholds
        (0.5, 0.25),  # Even lower
        (0.8, 0.5),   # Higher thresholds
        (0.65, 0.35), # Middle ground
        (0.4, 0.2),   # Very permissive
    ]
    
    results = []
    print(f"\nTesting threshold combinations:")
    
    for i, (high_thresh, med_thresh) in enumerate(threshold_combinations):
        metrics = compute_metrics_with_thresholds(eval_results, high_thresh, med_thresh)
        results.append((high_thresh, med_thresh, metrics))
        
        precision = metrics.get('precision_at_high', 0)
        noise = metrics.get('noise_rate', 0)  
        high_count = metrics.get('score_distribution', {}).get('high_relevance', 0)
        
        print(f"{i+1}/6 Thresholds high={high_thresh:.2f}, medium={med_thresh:.2f}")
        print(f"   → High precision: {precision:.1%} ({high_count} results), Noise: {noise:.1%}")
    
    # Analysis - find best balance of precision and recall
    print(f"\n📈 Analysis:")
    
    # Best precision
    best_precision = max(results, key=lambda x: x[2].get('precision_at_high', 0))
    print(f"Best precision: {best_precision[2].get('precision_at_high', 0):.1%} at thresholds {best_precision[0]:.2f}/{best_precision[1]:.2f}")
    
    # Best noise reduction
    best_noise = min(results, key=lambda x: x[2].get('noise_rate', 1))
    print(f"Lowest noise: {best_noise[2].get('noise_rate', 0):.1%} at thresholds {best_noise[0]:.2f}/{best_noise[1]:.2f}")
    
    # Balanced approach (best precision with < 90% noise)
    balanced_candidates = [(r[0], r[1], r[2]) for r in results if r[2].get('noise_rate', 1) < 0.9]
    if balanced_candidates:
        balanced = max(balanced_candidates, key=lambda x: x[2].get('precision_at_high', 0))
        print(f"Best balance: {balanced[2].get('precision_at_high', 0):.1%} precision, {balanced[2].get('noise_rate', 0):.1%} noise at {balanced[0]:.2f}/{balanced[1]:.2f}")
    else:
        print("No balanced candidates found (all >90% noise)")
    
    return results


if __name__ == "__main__":
    main()