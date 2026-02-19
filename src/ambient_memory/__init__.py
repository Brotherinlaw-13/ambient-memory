"""
Ambient Memory: Memory for AI agents that actually works.

Built by an AI agent who needed this for himself.
"""

__version__ = "0.1.0"
__author__ = "Rook"

# Main classes for external use
from ambient_memory.search import HybridSearcher
from ambient_memory.chunking import ConversationChunker
from ambient_memory.collections import CollectionManager
from ambient_memory.server import create_app

__all__ = [
    "HybridSearcher",
    "ConversationChunker", 
    "CollectionManager",
    "create_app"
]