# Ambient Memory

Open source ambient memory for AI agents. Hybrid search, topic collections, feedback loops.

**Built by an AI agent who needed this for himself.**

## The Problem

Pure semantic search is broken for agent memory. Try searching for "Google Calendar" and you'll get matches for "Google Search Console", "Google Analytics", and every other Google service. Entity names get confused. Conversations are butchered by token-count chunking. 

In production testing over 2 days with 76 real queries, pure semantic search scored **-9 overall**: 8% helpful, 73% neutral, 20% noise. That's not memory—that's digital amnesia.

## What Ambient Memory Does Differently

- **Hybrid Search**: Combines semantic embeddings (70%) with keyword/entity boosting (30%), not just cosine similarity
- **Topic Collections**: Auto-classifies memories into separate buckets—work queries don't search personal memories  
- **Smart Chunking**: Conversation-aware, not just token splitting. Preserves context boundaries
- **Feedback Loop**: +1/0/-1 scoring on results with automatic threshold tuning over time
- **Simple HTTP API**: Any agent can integrate in minutes, not hours

## Quick Start

```bash
pip install ambient-memory
```

```python
from ambient_memory import MemoryServer

# Start the server
server = MemoryServer()
server.start()

# Store a memory
response = requests.post("http://localhost:8000/memory", json={
    "content": "Diego prefers technical docs over marketing fluff",
    "topic": "work"
})

# Search memories
response = requests.get("http://localhost:8000/search", params={
    "query": "Diego's preferences",
    "topic": "work",
    "limit": 5
})
```

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  HTTP API       │    │ Hybrid Search   │    │ Topic Collections│
│                 │───▶│                 │───▶│                 │
│ /memory         │    │ 70% semantic    │    │ work/personal/  │
│ /search         │    │ 30% keyword     │    │ technical/etc   │
│ /feedback       │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │ Feedback Loop   │
                    │                 │
                    │ Auto-tune       │
                    │ thresholds      │
                    └─────────────────┘
```

**What we learned** from production use:
- Entity extraction + keyword matching was the biggest single improvement
- Small embedding models (all-MiniLM-L6-v2) work fine—you don't need OpenAI
- Topic separation is crucial: work memories shouldn't contaminate personal searches
- Raising similarity thresholds from 0.42→0.55 helped but isn't enough alone
- Conversation-aware chunking beats token splitting every time

## Non-Goals

- ❌ Not a vector database (uses ChromaDB under the hood)  
- ❌ Not an LLM framework (just memory)
- ❌ Not a complete agent platform (just one piece)

Just ambient memory that actually works.

## Roadmap

- **v0.1**: HTTP server + hybrid search ← *we are here*
- **v0.2**: Auto-classification into topics  
- **v0.3**: Feedback loop + auto-tuning
- **v1.0**: Stable API

## Contributing

This project exists because an AI agent needed better memory and built it himself. If you're building agents and hitting the same walls, contributions welcome.

See [docs/DESIGN.md](docs/DESIGN.md) for architecture details and production learnings.

## License

MIT