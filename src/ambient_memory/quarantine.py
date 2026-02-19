"""
Quarantine access tracking for ambient memory.

Tracks when quarantined memories are accessed (searched and returned to user)
to identify candidates for promotion to regular collections.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime, timezone


class QuarantineTracker:
    """
    Tracks access to quarantined memories for promotion decisions.
    
    Records every time a quarantine memory is accessed and provides
    methods to identify promotion candidates based on usage patterns.
    """
    
    def __init__(self, access_file: str = ".quarantine-access.json"):
        """
        Initialise quarantine tracker.
        
        Args:
            access_file: Path to JSON file storing access records
        """
        self.access_file = Path(access_file)
        self._access_data = self._load_access_data()
    
    def _load_access_data(self) -> Dict[str, Any]:
        """Load access data from JSON file."""
        if not self.access_file.exists():
            return {}
            
        try:
            with open(self.access_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    
    def _save_access_data(self) -> None:
        """Save access data to JSON file."""
        try:
            # Ensure parent directory exists
            self.access_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.access_file, 'w') as f:
                json.dump(self._access_data, f, indent=2)
        except IOError:
            # Non-critical error, continue without saving
            pass
    
    def record_access(self, memory_id: str) -> None:
        """
        Record an access to a quarantined memory.
        
        Args:
            memory_id: ID of the memory that was accessed
        """
        current_time = datetime.now(timezone.utc).isoformat()
        
        if memory_id not in self._access_data:
            self._access_data[memory_id] = {
                "access_count": 0,
                "first_access": current_time,
                "last_access": current_time,
                "access_history": []
            }
        
        # Update access record
        record = self._access_data[memory_id]
        record["access_count"] += 1
        record["last_access"] = current_time
        record["access_history"].append(current_time)
        
        # Keep only last 10 access timestamps to limit file size
        if len(record["access_history"]) > 10:
            record["access_history"] = record["access_history"][-10:]
        
        # Save immediately to persist the access
        self._save_access_data()
    
    def get_promotion_candidates(self) -> List[Dict[str, Any]]:
        """
        Get list of quarantine memories that are candidates for promotion.
        
        Promotion criteria:
        - 3+ accesses
        - First access 24+ hours ago
        - Not flagged
        
        Returns:
            List of candidate records with memory_id and access stats
        """
        candidates = []
        current_time = datetime.now(timezone.utc)
        
        for memory_id, record in self._access_data.items():
            # Check access count threshold
            if record["access_count"] < 3:
                continue
            
            # Check time threshold (24+ hours since first access)
            try:
                first_access_time = datetime.fromisoformat(
                    record["first_access"].replace('Z', '+00:00')
                )
                hours_since_first = (current_time - first_access_time).total_seconds() / 3600
                
                if hours_since_first < 24:
                    continue
            except (ValueError, TypeError):
                # Invalid timestamp, skip
                continue
            
            # Check if flagged (not implemented yet, but placeholder)
            if record.get("flagged", False):
                continue
            
            # This is a valid candidate
            candidates.append({
                "memory_id": memory_id,
                "access_count": record["access_count"],
                "first_access": record["first_access"],
                "last_access": record["last_access"],
                "hours_since_first": round(hours_since_first, 1),
                "days_since_first": round(hours_since_first / 24, 1)
            })
        
        # Sort by access count descending, then by days since first access
        candidates.sort(key=lambda x: (-x["access_count"], -x["days_since_first"]))
        
        return candidates
    
    def flag_memory(self, memory_id: str, reason: str = "") -> None:
        """
        Flag a memory to exclude it from promotion candidates.
        
        Args:
            memory_id: ID of the memory to flag
            reason: Optional reason for flagging
        """
        if memory_id in self._access_data:
            self._access_data[memory_id]["flagged"] = True
            self._access_data[memory_id]["flag_reason"] = reason
            self._access_data[memory_id]["flagged_at"] = datetime.now(timezone.utc).isoformat()
            self._save_access_data()
    
    def unflag_memory(self, memory_id: str) -> None:
        """
        Remove flag from a memory.
        
        Args:
            memory_id: ID of the memory to unflag
        """
        if memory_id in self._access_data:
            self._access_data[memory_id].pop("flagged", None)
            self._access_data[memory_id].pop("flag_reason", None)
            self._access_data[memory_id].pop("flagged_at", None)
            self._save_access_data()
    
    def get_access_stats(self, memory_id: str) -> Dict[str, Any]:
        """
        Get access statistics for a specific memory.
        
        Args:
            memory_id: ID of the memory
            
        Returns:
            Dictionary with access statistics, or None if not found
        """
        return self._access_data.get(memory_id)
    
    def clear_old_records(self, days_threshold: int = 90) -> int:
        """
        Clear access records older than the specified threshold.
        
        Args:
            days_threshold: Remove records with no access in this many days
            
        Returns:
            Number of records removed
        """
        current_time = datetime.now(timezone.utc)
        to_remove = []
        
        for memory_id, record in self._access_data.items():
            try:
                last_access_time = datetime.fromisoformat(
                    record["last_access"].replace('Z', '+00:00')
                )
                days_since_last = (current_time - last_access_time).total_seconds() / (24 * 3600)
                
                if days_since_last > days_threshold:
                    to_remove.append(memory_id)
            except (ValueError, TypeError):
                # Invalid timestamp, remove it
                to_remove.append(memory_id)
        
        # Remove old records
        for memory_id in to_remove:
            del self._access_data[memory_id]
        
        if to_remove:
            self._save_access_data()
        
        return len(to_remove)