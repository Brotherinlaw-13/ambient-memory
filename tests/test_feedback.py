"""
Tests for feedback-based threshold tuning functionality.

Tests the system that adjusts search thresholds per collection
based on user feedback scores.
"""

import json
import pytest
import tempfile
from pathlib import Path
from ambient_memory.search import HybridSearcher


class TestFeedbackThresholdTuning:
    """Test cases for feedback-based threshold adjustment."""
    
    def test_load_empty_feedback(self):
        """Test loading when no feedback file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                # Change to temp directory
                import os
                os.chdir(temp_dir)
                
                searcher = HybridSearcher(chroma_path=f"{temp_dir}/.chromadb")
                adjustments = searcher._load_threshold_adjustments()
                
                # Should return empty dict when no feedback file
                assert adjustments == {}
                
            finally:
                os.chdir(original_cwd)
    
    def test_load_feedback_poor_performance(self):
        """Test threshold adjustment for poor performing collections."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                import os
                os.chdir(temp_dir)
                
                # Create feedback file with poor scores
                feedback_data = [
                    {"query": "test1", "result_text": "result1", "score": -1, "collection": "memory_work"},
                    {"query": "test2", "result_text": "result2", "score": -1, "collection": "memory_work"},
                    {"query": "test3", "result_text": "result3", "score": 0, "collection": "memory_work"},
                    {"query": "test4", "result_text": "result4", "score": -1, "collection": "memory_projects"},
                    {"query": "test5", "result_text": "result5", "score": -1, "collection": "memory_projects"},
                ]
                
                with open(".ambient-memory-feedback.jsonl", "w") as f:
                    for entry in feedback_data:
                        f.write(json.dumps(entry) + "\n")
                
                searcher = HybridSearcher(chroma_path=f"{temp_dir}/.chromadb")
                adjustments = searcher._load_threshold_adjustments()
                
                # memory_work average: (-1 + -1 + 0) / 3 = -0.67 (< -0.3, so +0.05)
                # memory_projects average: (-1 + -1) / 2 = -1.0 (< -0.3, so +0.05)
                assert adjustments["memory_work"] == 0.05
                assert adjustments["memory_projects"] == 0.05
                
            finally:
                os.chdir(original_cwd)
    
    def test_load_feedback_good_performance(self):
        """Test threshold adjustment for well-performing collections."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                import os
                os.chdir(temp_dir)
                
                # Create feedback file with good scores
                feedback_data = [
                    {"query": "test1", "result_text": "result1", "score": 1, "collection": "memory_personal"},
                    {"query": "test2", "result_text": "result2", "score": 1, "collection": "memory_personal"},
                    {"query": "test3", "result_text": "result3", "score": 0, "collection": "memory_personal"},
                    {"query": "test4", "result_text": "result4", "score": 1, "collection": "memory_general"},
                    {"query": "test5", "result_text": "result5", "score": 1, "collection": "memory_general"},
                ]
                
                with open(".ambient-memory-feedback.jsonl", "w") as f:
                    for entry in feedback_data:
                        f.write(json.dumps(entry) + "\n")
                
                searcher = HybridSearcher(chroma_path=f"{temp_dir}/.chromadb")
                adjustments = searcher._load_threshold_adjustments()
                
                # memory_personal average: (1 + 1 + 0) / 3 = 0.67 (> 0.3, so -0.05)
                # memory_general average: (1 + 1) / 2 = 1.0 (> 0.3, so -0.05)
                assert adjustments["memory_personal"] == -0.05
                assert adjustments["memory_general"] == -0.05
                
            finally:
                os.chdir(original_cwd)
    
    def test_load_feedback_neutral_performance(self):
        """Test no adjustment for neutral performing collections."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                import os
                os.chdir(temp_dir)
                
                # Create feedback file with neutral scores
                feedback_data = [
                    {"query": "test1", "result_text": "result1", "score": 0, "collection": "memory_work"},
                    {"query": "test2", "result_text": "result2", "score": 1, "collection": "memory_work"},
                    {"query": "test3", "result_text": "result3", "score": -1, "collection": "memory_work"},
                    {"query": "test4", "result_text": "result4", "score": 0, "collection": "memory_work"},
                ]
                
                with open(".ambient-memory-feedback.jsonl", "w") as f:
                    for entry in feedback_data:
                        f.write(json.dumps(entry) + "\n")
                
                searcher = HybridSearcher(chroma_path=f"{temp_dir}/.chromadb")
                adjustments = searcher._load_threshold_adjustments()
                
                # memory_work average: (0 + 1 + -1 + 0) / 4 = 0.0 (neutral, no adjustment)
                assert adjustments == {}
                
            finally:
                os.chdir(original_cwd)
    
    def test_get_effective_threshold_default(self):
        """Test effective threshold calculation with default settings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            searcher = HybridSearcher(
                chroma_path=f"{temp_dir}/.chromadb",
                min_similarity_threshold=0.6
            )
            searcher.threshold_adjustments = {}
            
            # Should return base threshold when no adjustments
            assert searcher.get_effective_threshold("memory_work") == 0.6
            assert searcher.get_effective_threshold("unknown_collection") == 0.6
    
    def test_get_effective_threshold_with_adjustments(self):
        """Test effective threshold calculation with adjustments."""
        with tempfile.TemporaryDirectory() as temp_dir:
            searcher = HybridSearcher(
                chroma_path=f"{temp_dir}/.chromadb",
                min_similarity_threshold=0.6
            )
            searcher.threshold_adjustments = {
                "memory_work": 0.05,  # Poor performance, raise threshold
                "memory_personal": -0.05,  # Good performance, lower threshold
            }
            
            # Should apply adjustments (with small floating point tolerance)
            assert abs(searcher.get_effective_threshold("memory_work") - 0.65) < 0.0001
            assert abs(searcher.get_effective_threshold("memory_personal") - 0.55) < 0.0001
            assert abs(searcher.get_effective_threshold("memory_general") - 0.6) < 0.0001  # No adjustment
    
    def test_get_effective_threshold_clamping(self):
        """Test that effective thresholds are clamped between 0.45 and 0.75."""
        with tempfile.TemporaryDirectory() as temp_dir:
            searcher = HybridSearcher(
                chroma_path=f"{temp_dir}/.chromadb",
                min_similarity_threshold=0.4  # Below minimum
            )
            searcher.threshold_adjustments = {
                "too_low": -0.1,  # Would be 0.3, but should clamp to 0.45
                "too_high": 0.5,  # Would be 0.9, but should clamp to 0.75
            }
            
            # Should clamp to valid range
            assert searcher.get_effective_threshold("too_low") == 0.45
            assert searcher.get_effective_threshold("too_high") == 0.75
            
            # Test with high base threshold
            searcher.min_similarity_threshold = 0.8
            searcher.threshold_adjustments = {"normal": 0.0}
            assert searcher.get_effective_threshold("normal") == 0.75  # Clamped down
    
    def test_malformed_feedback_data(self):
        """Test handling of malformed feedback data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                import os
                os.chdir(temp_dir)
                
                # Create feedback file with malformed data
                with open(".ambient-memory-feedback.jsonl", "w") as f:
                    f.write('{"query": "test1", "score": 1}\n')  # Missing collection
                    f.write('invalid json line\n')  # Invalid JSON
                    f.write('{"query": "test2", "result_text": "result2", "score": "not_a_number", "collection": "memory_work"}\n')  # Invalid score
                    f.write('{"query": "test3", "result_text": "result3", "score": 0, "collection": "memory_work"}\n')  # Valid entry with neutral score
                
                searcher = HybridSearcher(chroma_path=f"{temp_dir}/.chromadb")
                adjustments = searcher._load_threshold_adjustments()
                
                # Should only process the valid entry with neutral score (0), so no adjustment
                assert adjustments == {}
                
            finally:
                os.chdir(original_cwd)
    
    def test_mixed_performance_collections(self):
        """Test collections with mixed feedback across different performance levels."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                import os
                os.chdir(temp_dir)
                
                feedback_data = [
                    # Good performing collection
                    {"query": "test1", "result_text": "result1", "score": 1, "collection": "good_collection"},
                    {"query": "test2", "result_text": "result2", "score": 1, "collection": "good_collection"},
                    {"query": "test3", "result_text": "result3", "score": 1, "collection": "good_collection"},
                    # Poor performing collection  
                    {"query": "test4", "result_text": "result4", "score": -1, "collection": "poor_collection"},
                    {"query": "test5", "result_text": "result5", "score": -1, "collection": "poor_collection"},
                    {"query": "test6", "result_text": "result6", "score": -1, "collection": "poor_collection"},
                    # Neutral collection
                    {"query": "test7", "result_text": "result7", "score": 0, "collection": "neutral_collection"},
                    {"query": "test8", "result_text": "result8", "score": 0, "collection": "neutral_collection"},
                    {"query": "test9", "result_text": "result9", "score": 0, "collection": "neutral_collection"},
                ]
                
                with open(".ambient-memory-feedback.jsonl", "w") as f:
                    for entry in feedback_data:
                        f.write(json.dumps(entry) + "\n")
                
                searcher = HybridSearcher(chroma_path=f"{temp_dir}/.chromadb")
                adjustments = searcher._load_threshold_adjustments()
                
                # good_collection: average 1.0 > 0.3, so -0.05
                # poor_collection: average -1.0 < -0.3, so +0.05
                # neutral_collection: average 0.0 in neutral range, no adjustment
                expected = {
                    "good_collection": -0.05,
                    "poor_collection": 0.05
                }
                assert adjustments == expected
                
            finally:
                os.chdir(original_cwd)
    
    def test_threshold_boundary_conditions(self):
        """Test feedback scores exactly at boundary conditions."""
        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                import os
                os.chdir(temp_dir)
                
                feedback_data = [
                    # Exactly at -0.3 boundary (should trigger adjustment)
                    {"query": "test1", "result_text": "result1", "score": -1, "collection": "boundary_low"},
                    {"query": "test2", "result_text": "result2", "score": 0, "collection": "boundary_low"},
                    {"query": "test3", "result_text": "result3", "score": 0, "collection": "boundary_low"},
                    {"query": "test4", "result_text": "result4", "score": 0, "collection": "boundary_low"},
                    {"query": "test5", "result_text": "result5", "score": 0, "collection": "boundary_low"},  # average = -0.2, no adjustment
                    # Exactly at +0.3 boundary
                    {"query": "test6", "result_text": "result6", "score": 1, "collection": "boundary_high"},
                    {"query": "test7", "result_text": "result7", "score": 0, "collection": "boundary_high"},
                    {"query": "test8", "result_text": "result8", "score": 0, "collection": "boundary_high"},
                    {"query": "test9", "result_text": "result9", "score": 0, "collection": "boundary_high"},
                    {"query": "test10", "result_text": "result10", "score": 0, "collection": "boundary_high"},  # average = 0.2, no adjustment
                ]
                
                with open(".ambient-memory-feedback.jsonl", "w") as f:
                    for entry in feedback_data:
                        f.write(json.dumps(entry) + "\n")
                
                searcher = HybridSearcher(chroma_path=f"{temp_dir}/.chromadb")
                adjustments = searcher._load_threshold_adjustments()
                
                # Both should be in neutral range, no adjustments
                assert adjustments == {}
                
            finally:
                os.chdir(original_cwd)