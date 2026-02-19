#!/usr/bin/env python3
"""Quick test of Phase 3 scoring logic without network calls."""

import sys
import json
from pathlib import Path
from datetime import datetime
sys.path.append('/Users/rook/workspace/ambient-memory/scripts')

from simple_phase3_eval import MessageExtractor, Phase3Scorer, Query

def main():
    print("🧠 Quick Phase 3 Scoring Test")
    print("=" * 30)
    
    # Extract queries
    telegram_dir = Path("/Users/rook/workspace/memory/telegram")
    extractor = MessageExtractor()
    scorer = Phase3Scorer()
    
    all_queries = []
    for day in range(10, 18):  # Feb 10-17
        file_path = telegram_dir / f"the-factory-2026-02-{day:02d}.md"
        if file_path.exists():
            queries = extractor.extract_from_file(file_path)
            print(f"{file_path.name}: {len(queries)} messages")
            all_queries.extend(queries)
    
    print(f"\n✅ Total extracted: {len(all_queries)} queries")
    
    # Test scoring with mock results
    print("\n🔍 Testing Phase 3 scoring logic...")
    
    all_scores = []
    high_rel = 0  # >= 0.4
    med_rel = 0   # 0.2-0.4
    low_rel = 0   # < 0.2
    
    # Mock search results for testing
    mock_results = [
        "This is about Darwin project development with AI features and database optimization.",
        "Discussion about Chowdown app restaurant recommendations and user interface improvements.", 
        "General conversation about daily tasks and scheduling meetings for the team.",
        "Random chatter about weather and morning routines with no specific context.",
    ]
    
    # Test each query against mock results
    for i, query in enumerate(all_queries):
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(all_queries)}")
        
        # Score query against each mock result
        for result in mock_results:
            score = scorer.score_relevance(query, result)
            all_scores.append(score)
            
            if score >= 0.4:
                high_rel += 1
            elif score >= 0.2:
                med_rel += 1
            else:
                low_rel += 1
    
    # Calculate metrics
    total_results = len(all_scores)
    avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
    precision = high_rel / total_results if total_results > 0 else 0
    noise_rate = low_rel / total_results if total_results > 0 else 0
    
    print("\n" + "=" * 40)
    print("📊 PHASE 3 SCORING TEST RESULTS")
    print("=" * 40)
    print(f"Total Queries:         {len(all_queries)}")
    print(f"Total Results:         {total_results}")
    print(f"Avg Relevance Score:   {avg_score:.3f}")
    print(f"High Precision:        {precision:.2%}")
    print(f"Noise Rate:            {noise_rate:.2%}")
    print(f"Coverage:              100.0%")  # All queries tested
    
    print(f"\nScore Distribution:")
    print(f"  High relevance (≥0.4): {high_rel} ({high_rel/total_results:.1%})")
    print(f"  Medium relevance:       {med_rel} ({med_rel/total_results:.1%})")
    print(f"  Low/noise (<0.2):       {low_rel} ({low_rel/total_results:.1%})")
    
    # Show sample queries and scores
    print(f"\n📋 Sample Queries:")
    for i, query in enumerate(all_queries[:5]):
        print(f"{i+1}. {query.text[:80]}...")
        for j, result in enumerate(mock_results[:2]):  # Show 2 results per query
            score = scorer.score_relevance(query, result)
            print(f"   Result {j+1} score: {score:.3f} - {result[:50]}...")
    
    # Save results
    eval_dir = Path("eval")
    eval_dir.mkdir(exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_type": "phase3_scoring_test",
        "total_queries": len(all_queries),
        "total_results": total_results,
        "metrics": {
            "avg_relevance_score": avg_score,
            "precision_at_high": precision,
            "noise_rate": noise_rate,
            "coverage": 1.0
        },
        "score_distribution": {
            "high_relevance": high_rel,
            "medium_relevance": med_rel,  
            "low_relevance": low_rel
        }
    }
    
    results_file = eval_dir / "phase6-scoring-test.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("\n✅ Phase 3 scoring test complete!")
    
    return results

if __name__ == "__main__":
    main()