"""
Smart chunking for conversational text that preserves context boundaries.

Designed for agent memory: detects speaker turns, timestamps, and topic shifts
to avoid cutting conversations mid-flow.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Chunk:
    """A chunk of text with metadata."""
    text: str
    start_line: int
    end_line: int
    speakers: List[str]  # Speakers present in this chunk
    timestamp: Optional[datetime] = None
    topic_hints: List[str] = None  # Keywords that might indicate topic
    
    def __post_init__(self):
        if self.topic_hints is None:
            self.topic_hints = []


class ConversationChunker:
    """
    Conversation-aware text chunker that preserves context boundaries.
    
    Features:
    - Detects speaker turns and keeps conversations together
    - Respects timestamp boundaries (don't split across days/sessions)
    - Identifies topic shifts using conversation patterns
    - Configurable chunk size with overlap for context preservation
    """
    
    def __init__(
        self,
        target_chunk_size: int = 1000,
        max_chunk_size: int = 1500,
        overlap_size: int = 100,
        min_chunk_size: int = 200
    ):
        """
        Initialise conversation chunker.
        
        Args:
            target_chunk_size: Target number of characters per chunk
            max_chunk_size: Maximum characters before forcing a split
            overlap_size: Characters to overlap between adjacent chunks
            min_chunk_size: Minimum chunk size (avoid tiny fragments)
        """
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        
        # Common patterns for detecting structure
        self.timestamp_patterns = [
            r'\[\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}',  # [2024-02-19 08:21] or [2024-02-19T08:21]
            r'\d{2}:\d{2}\s*[AP]M',  # 08:21 AM
            r'\d{1,2}/\d{1,2}/\d{4}',  # 2/19/2024
            r'\w+ \d{1,2}, \d{4}',  # February 19, 2024
        ]
        
        self.speaker_patterns = [
            r'^([A-Z][a-zA-Z\s]+?):\s',  # "Diego: message"
            r'^\*\*([A-Z][a-zA-Z\s]+?)\*\*',  # "**Diego**"
            r'^<([^>]+)>\s',  # "<Diego> message"
            r'^@([a-zA-Z_]+)\s',  # "@diego message"
        ]
        
        # Topic shift indicators
        self.topic_shift_patterns = [
            r'(?i)^(anyway|so|btw|by the way|speaking of|on another note)',
            r'(?i)^(new topic|different topic|changing subject)',
            r'(?i)^(quick question|one more thing|also)',
        ]
    
    def _detect_timestamp(self, line: str) -> Optional[datetime]:
        """Try to extract a timestamp from a line."""
        for pattern in self.timestamp_patterns:
            match = re.search(pattern, line)
            if match:
                timestamp_str = match.group(0)
                # Try common formats
                try:
                    # ISO-like format
                    if 'T' in timestamp_str or '-' in timestamp_str:
                        return datetime.fromisoformat(timestamp_str.replace('[', '').replace(']', ''))
                except:
                    pass
                # For now, return None if we can't parse
                # Could extend this with more format handling
        return None
    
    def _detect_speaker(self, line: str) -> Optional[str]:
        """Try to extract speaker name from a line."""
        for pattern in self.speaker_patterns:
            match = re.match(pattern, line.strip())
            if match:
                return match.group(1).strip()
        return None
    
    def _is_topic_shift(self, line: str) -> bool:
        """Check if line indicates a topic shift."""
        line_lower = line.lower().strip()
        for pattern in self.topic_shift_patterns:
            if re.match(pattern, line_lower):
                return True
        return False
    
    def _extract_topic_hints(self, text: str) -> List[str]:
        """Extract potential topic keywords from text."""
        # Simple approach: capitalised words and common project/tool names
        hints = set()
        
        # Capitalised words (potential proper nouns, tools, projects)
        words = re.findall(r'\b[A-Z][a-zA-Z]+\b', text)
        for word in words:
            if len(word) >= 3 and word not in ['The', 'This', 'That', 'They', 'Then', 'There']:
                hints.add(word)
        
        # ALL-CAPS words (acronyms, tool names)
        caps_words = re.findall(r'\b[A-Z]{2,}\b', text)
        hints.update(caps_words)
        
        return list(hints)[:10]  # Limit to top 10 hints
    
    def chunk_text(self, text: str, source_name: str = "") -> List[Chunk]:
        """
        Chunk text while preserving conversation boundaries.
        
        Args:
            text: Input text to chunk
            source_name: Source identifier for debugging
        
        Returns:
            List of Chunk objects
        """
        lines = text.split('\n')
        chunks = []
        current_chunk_lines = []
        current_chunk_chars = 0
        current_speakers = set()
        current_timestamp = None
        chunk_start_line = 0
        
        for i, line in enumerate(lines):
            line_chars = len(line)
            
            # Detect structural elements
            line_timestamp = self._detect_timestamp(line)
            line_speaker = self._detect_speaker(line)
            is_topic_shift = self._is_topic_shift(line)
            
            # Check if we should force a chunk boundary
            should_split = False
            
            # Force split if we hit max size
            if current_chunk_chars + line_chars > self.max_chunk_size:
                should_split = True
            
            # Split on significant time gaps (more than 1 hour)
            elif (line_timestamp and current_timestamp and 
                  abs((line_timestamp - current_timestamp).total_seconds()) > 3600):
                should_split = True
            
            # Split on topic shifts if chunk is reasonably sized
            elif is_topic_shift and current_chunk_chars > self.min_chunk_size:
                should_split = True
            
            # Split when we hit target size and find a good boundary
            elif (current_chunk_chars > self.target_chunk_size and 
                  (line_speaker or line_timestamp or len(line.strip()) == 0)):
                should_split = True
            
            # If we should split and have content, create chunk
            if should_split and current_chunk_lines:
                chunk_text = '\n'.join(current_chunk_lines)
                if len(chunk_text.strip()) >= self.min_chunk_size:
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_line=chunk_start_line,
                        end_line=chunk_start_line + len(current_chunk_lines) - 1,
                        speakers=list(current_speakers),
                        timestamp=current_timestamp,
                        topic_hints=self._extract_topic_hints(chunk_text)
                    ))
                
                # Start new chunk with overlap
                overlap_lines = []
                overlap_chars = 0
                
                # Add last few lines as overlap if they fit
                for j in range(len(current_chunk_lines) - 1, -1, -1):
                    if overlap_chars + len(current_chunk_lines[j]) <= self.overlap_size:
                        overlap_lines.insert(0, current_chunk_lines[j])
                        overlap_chars += len(current_chunk_lines[j])
                    else:
                        break
                
                current_chunk_lines = overlap_lines
                current_chunk_chars = overlap_chars
                current_speakers = set()
                chunk_start_line = i
            
            # Add current line to chunk
            current_chunk_lines.append(line)
            current_chunk_chars += line_chars + 1  # +1 for newline
            
            # Update tracking variables
            if line_speaker:
                current_speakers.add(line_speaker)
            if line_timestamp:
                current_timestamp = line_timestamp
        
        # Handle final chunk
        if current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines)
            if len(chunk_text.strip()) >= self.min_chunk_size:
                chunks.append(Chunk(
                    text=chunk_text,
                    start_line=chunk_start_line,
                    end_line=chunk_start_line + len(current_chunk_lines) - 1,
                    speakers=list(current_speakers),
                    timestamp=current_timestamp,
                    topic_hints=self._extract_topic_hints(chunk_text)
                ))
        
        return chunks
    
    def chunk_conversation_file(self, file_path: str) -> List[Chunk]:
        """Chunk a conversation file while preserving structure."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return self.chunk_text(content, source_name=file_path)