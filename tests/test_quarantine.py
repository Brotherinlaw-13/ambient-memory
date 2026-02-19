"""
Tests for quarantine access tracking functionality.

Tests the system that tracks access to quarantined memories
and identifies promotion candidates.
"""

import json
import tempfile
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from ambient_memory.quarantine import QuarantineTracker


class TestQuarantineTracker:
    """Test cases for quarantine access tracking."""
    
    def test_init_no_existing_file(self):
        """Test initialisation when no access file exists."""
        with tempfile.TemporaryDirectory() as temp_dir:
            access_file = f"{temp_dir}/test-access.json"
            tracker = QuarantineTracker(access_file=access_file)
            
            # Should start with empty access data
            assert tracker._access_data == {}
            assert not Path(access_file).exists()
    
    def test_init_with_existing_file(self):
        """Test initialisation with existing access file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            access_file = f"{temp_dir}/test-access.json"
            
            # Create initial data
            initial_data = {
                "memory123": {
                    "access_count": 2,
                    "first_access": "2024-01-01T10:00:00+00:00",
                    "last_access": "2024-01-02T10:00:00+00:00",
                    "access_history": ["2024-01-01T10:00:00+00:00", "2024-01-02T10:00:00+00:00"]
                }
            }
            
            Path(access_file).write_text(json.dumps(initial_data))
            
            tracker = QuarantineTracker(access_file=access_file)
            assert tracker._access_data == initial_data
    
    def test_init_malformed_file(self):
        """Test initialisation with malformed JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            access_file = f"{temp_dir}/test-access.json"
            
            # Create malformed JSON
            Path(access_file).write_text("invalid json content")
            
            tracker = QuarantineTracker(access_file=access_file)
            # Should start with empty data when file is malformed
            assert tracker._access_data == {}
    
    def test_record_access_new_memory(self):
        """Test recording access for a new memory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            access_file = f"{temp_dir}/test-access.json"
            tracker = QuarantineTracker(access_file=access_file)
            
            memory_id = "memory123"
            tracker.record_access(memory_id)
            
            # Should create new record
            assert memory_id in tracker._access_data
            record = tracker._access_data[memory_id]
            assert record["access_count"] == 1
            assert record["first_access"] == record["last_access"]
            assert len(record["access_history"]) == 1
            
            # File should be saved
            assert Path(access_file).exists()
    
    def test_record_access_existing_memory(self):
        """Test recording access for an existing memory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            access_file = f"{temp_dir}/test-access.json"
            tracker = QuarantineTracker(access_file=access_file)
            
            memory_id = "memory123"
            
            # Record first access
            tracker.record_access(memory_id)
            first_access = tracker._access_data[memory_id]["first_access"]
            
            # Record second access
            tracker.record_access(memory_id)
            record = tracker._access_data[memory_id]
            
            # Should update existing record
            assert record["access_count"] == 2
            assert record["first_access"] == first_access  # Should not change
            assert record["last_access"] != first_access  # Should be updated
            assert len(record["access_history"]) == 2
    
    def test_record_access_history_limit(self):
        """Test that access history is limited to last 10 entries."""
        with tempfile.TemporaryDirectory() as temp_dir:
            access_file = f"{temp_dir}/test-access.json"
            tracker = QuarantineTracker(access_file=access_file)
            
            memory_id = "memory123"
            
            # Record 15 accesses
            for i in range(15):
                tracker.record_access(memory_id)
            
            record = tracker._access_data[memory_id]
            assert record["access_count"] == 15
            assert len(record["access_history"]) == 10  # Should be limited to 10
    
    def test_get_promotion_candidates_empty(self):
        """Test getting promotion candidates when no accesses recorded."""
        with tempfile.TemporaryDirectory() as temp_dir:
            access_file = f"{temp_dir}/test-access.json"
            tracker = QuarantineTracker(access_file=access_file)
            
            candidates = tracker.get_promotion_candidates()
            assert candidates == []
    
    def test_get_promotion_candidates_insufficient_access(self):
        """Test that memories with < 3 accesses are not promoted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            access_file = f"{temp_dir}/test-access.json"
            tracker = QuarantineTracker(access_file=access_file)
            
            # Create record with only 2 accesses
            yesterday = datetime.now(timezone.utc) - timedelta(days=1)
            tracker._access_data["memory123"] = {
                "access_count": 2,
                "first_access": yesterday.isoformat(),
                "last_access": datetime.now(timezone.utc).isoformat(),
                "access_history": [yesterday.isoformat(), datetime.now(timezone.utc).isoformat()]
            }
            
            candidates = tracker.get_promotion_candidates()
            assert candidates == []
    
    def test_get_promotion_candidates_insufficient_time(self):
        """Test that memories accessed < 24h ago are not promoted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            access_file = f"{temp_dir}/test-access.json"
            tracker = QuarantineTracker(access_file=access_file)
            
            # Create record with sufficient accesses but recent first access
            recent_time = datetime.now(timezone.utc) - timedelta(hours=12)  # Only 12 hours ago
            tracker._access_data["memory123"] = {
                "access_count": 5,
                "first_access": recent_time.isoformat(),
                "last_access": datetime.now(timezone.utc).isoformat(),
                "access_history": [recent_time.isoformat()]
            }
            
            candidates = tracker.get_promotion_candidates()
            assert candidates == []
    
    def test_get_promotion_candidates_flagged(self):
        """Test that flagged memories are not promoted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            access_file = f"{temp_dir}/test-access.json"
            tracker = QuarantineTracker(access_file=access_file)
            
            # Create record that meets criteria but is flagged
            old_time = datetime.now(timezone.utc) - timedelta(days=2)
            tracker._access_data["memory123"] = {
                "access_count": 5,
                "first_access": old_time.isoformat(),
                "last_access": datetime.now(timezone.utc).isoformat(),
                "access_history": [old_time.isoformat()],
                "flagged": True
            }
            
            candidates = tracker.get_promotion_candidates()
            assert candidates == []
    
    def test_get_promotion_candidates_valid(self):
        """Test getting valid promotion candidates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            access_file = f"{temp_dir}/test-access.json"
            tracker = QuarantineTracker(access_file=access_file)
            
            # Create multiple records with different characteristics
            base_time = datetime.now(timezone.utc)
            
            # Valid candidate 1
            old_time_1 = base_time - timedelta(days=3)
            tracker._access_data["memory123"] = {
                "access_count": 5,
                "first_access": old_time_1.isoformat(),
                "last_access": base_time.isoformat(),
                "access_history": [old_time_1.isoformat()]
            }
            
            # Valid candidate 2 (more accesses, should be ranked higher)
            old_time_2 = base_time - timedelta(days=2)
            tracker._access_data["memory456"] = {
                "access_count": 8,
                "first_access": old_time_2.isoformat(),
                "last_access": base_time.isoformat(),
                "access_history": [old_time_2.isoformat()]
            }
            
            # Invalid candidate (too recent)
            recent_time = base_time - timedelta(hours=12)
            tracker._access_data["memory789"] = {
                "access_count": 10,
                "first_access": recent_time.isoformat(),
                "last_access": base_time.isoformat(),
                "access_history": [recent_time.isoformat()]
            }
            
            candidates = tracker.get_promotion_candidates()
            
            assert len(candidates) == 2
            
            # Should be sorted by access count (higher first)
            assert candidates[0]["memory_id"] == "memory456"
            assert candidates[0]["access_count"] == 8
            assert candidates[1]["memory_id"] == "memory123"
            assert candidates[1]["access_count"] == 5
            
            # Should include calculated time metrics
            assert "hours_since_first" in candidates[0]
            assert "days_since_first" in candidates[0]
            assert candidates[0]["hours_since_first"] >= 48  # At least 2 days
    
    def test_flag_memory(self):
        """Test flagging a memory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            access_file = f"{temp_dir}/test-access.json"
            tracker = QuarantineTracker(access_file=access_file)
            
            memory_id = "memory123"
            tracker.record_access(memory_id)
            
            # Flag the memory
            reason = "inappropriate content"
            tracker.flag_memory(memory_id, reason)
            
            record = tracker._access_data[memory_id]
            assert record["flagged"] is True
            assert record["flag_reason"] == reason
            assert "flagged_at" in record
    
    def test_unflag_memory(self):
        """Test unflagging a memory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            access_file = f"{temp_dir}/test-access.json"
            tracker = QuarantineTracker(access_file=access_file)
            
            memory_id = "memory123"
            tracker.record_access(memory_id)
            tracker.flag_memory(memory_id, "test reason")
            
            # Unflag the memory
            tracker.unflag_memory(memory_id)
            
            record = tracker._access_data[memory_id]
            assert "flagged" not in record
            assert "flag_reason" not in record
            assert "flagged_at" not in record
    
    def test_get_access_stats(self):
        """Test getting access statistics for a memory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            access_file = f"{temp_dir}/test-access.json"
            tracker = QuarantineTracker(access_file=access_file)
            
            memory_id = "memory123"
            tracker.record_access(memory_id)
            tracker.record_access(memory_id)
            
            stats = tracker.get_access_stats(memory_id)
            assert stats is not None
            assert stats["access_count"] == 2
            assert "first_access" in stats
            assert "last_access" in stats
            
            # Non-existent memory should return None
            assert tracker.get_access_stats("nonexistent") is None
    
    def test_clear_old_records(self):
        """Test clearing old access records."""
        with tempfile.TemporaryDirectory() as temp_dir:
            access_file = f"{temp_dir}/test-access.json"
            tracker = QuarantineTracker(access_file=access_file)
            
            base_time = datetime.now(timezone.utc)
            
            # Create old record (should be removed)
            old_time = base_time - timedelta(days=100)
            tracker._access_data["old_memory"] = {
                "access_count": 3,
                "first_access": old_time.isoformat(),
                "last_access": old_time.isoformat(),
                "access_history": [old_time.isoformat()]
            }
            
            # Create recent record (should be kept)
            recent_time = base_time - timedelta(days=30)
            tracker._access_data["recent_memory"] = {
                "access_count": 3,
                "first_access": recent_time.isoformat(),
                "last_access": recent_time.isoformat(),
                "access_history": [recent_time.isoformat()]
            }
            
            # Clear records older than 90 days
            removed_count = tracker.clear_old_records(days_threshold=90)
            
            assert removed_count == 1
            assert "old_memory" not in tracker._access_data
            assert "recent_memory" in tracker._access_data
    
    def test_clear_old_records_malformed_timestamps(self):
        """Test clearing records with malformed timestamps."""
        with tempfile.TemporaryDirectory() as temp_dir:
            access_file = f"{temp_dir}/test-access.json"
            tracker = QuarantineTracker(access_file=access_file)
            
            # Create record with invalid timestamp
            tracker._access_data["malformed_memory"] = {
                "access_count": 3,
                "first_access": "invalid-timestamp",
                "last_access": "invalid-timestamp",
                "access_history": ["invalid-timestamp"]
            }
            
            # Should remove records with malformed timestamps
            removed_count = tracker.clear_old_records()
            assert removed_count == 1
            assert "malformed_memory" not in tracker._access_data
    
    def test_candidate_sorting(self):
        """Test that promotion candidates are sorted correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            access_file = f"{temp_dir}/test-access.json"
            tracker = QuarantineTracker(access_file=access_file)
            
            base_time = datetime.now(timezone.utc)
            
            # Create candidates with different access counts and ages
            candidates_data = [
                ("memory1", 3, 2),  # 3 accesses, 2 days old
                ("memory2", 5, 3),  # 5 accesses, 3 days old  
                ("memory3", 5, 0.5),  # 5 accesses, 12 hours old - should be excluded (too recent)
                ("memory4", 8, 2),  # 8 accesses, 2 days old
                ("memory5", 3, 5),  # 3 accesses, 5 days old
            ]
            
            for memory_id, access_count, days_old in candidates_data:
                first_access = base_time - timedelta(days=days_old)
                tracker._access_data[memory_id] = {
                    "access_count": access_count,
                    "first_access": first_access.isoformat(),
                    "last_access": base_time.isoformat(),
                    "access_history": [first_access.isoformat()]
                }
            
            candidates = tracker.get_promotion_candidates()
            
            # Should exclude memory3 (too recent)
            memory_ids = [c["memory_id"] for c in candidates]
            assert "memory3" not in memory_ids
            assert len(candidates) == 4
            
            # Should be sorted by access count descending, then by age (older first for same access count)
            # Expected order: memory4 (8 accesses), memory2 (5 accesses), memory5 (3 accesses, 5 days), memory1 (3 accesses, 2 days)
            expected_order = ["memory4", "memory2", "memory5", "memory1"]
            actual_order = [c["memory_id"] for c in candidates]
            assert actual_order == expected_order