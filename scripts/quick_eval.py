#!/usr/bin/env python3
"""
Quick evaluation script for fast iteration testing.

Uses a subset of queries for faster feedback during iteration.
"""

import sys
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the full evaluation classes
from evaluate import MemoryEvaluator, ConversationParser, RelevanceScorer


def main():
    print("🚀 Quick Evaluation (Subset)")
    print("=" * 40)
    
    # Set up paths  
    workspace = Path(__file__).parent.parent
    telegram_dir = Path("/Users/rook/workspace/memory/telegram")
    
    # Use just 2 files for quick testing
    test_files = [
        telegram_dir / "the-factory-2026-02-16.md",
        telegram_dir / "the-factory-2026-02-17.md"
    ]
    
    available_files = [f for f in test_files if f.exists()]
    if not available_files:
        print("❌ No test files found")
        return
    
    print(f"📁 Quick test with: {[f.name for f in available_files]}")
    
    # Initialize evaluator
    evaluator = MemoryEvaluator()
    
    # Extract queries
    all_queries = []
    for file_path in available_files:
        queries = evaluator.parser.extract_diego_messages(file_path)
        print(f"   {file_path.name}: {len(queries)} messages")
        all_queries.extend(queries)
    
    # Limit to first 30 queries for quick testing
    test_queries = all_queries[:30]
    print(f"✅ Testing with {len(test_queries)} queries")
    
    # Run evaluation
    eval_results = evaluator.evaluate_queries(test_queries)
    
    # Compute metrics
    metrics = evaluator.compute_metrics(eval_results)
    
    # Display results
    print(f"\n📊 QUICK RESULTS")
    print(f"Coverage:          {metrics.get('coverage', 0):.2%}")
    print(f"Avg Relevance:     {metrics.get('avg_relevance_score', 0):.3f}")
    print(f"High Precision:    {metrics.get('precision_at_high', 0):.2%}")
    print(f"Noise Rate:        {metrics.get('noise_rate', 0):.2%}")
    print(f"Avg Time:          {metrics.get('avg_execution_time_ms', 0):.0f}ms")
    
    return metrics


if __name__ == "__main__":
    main()