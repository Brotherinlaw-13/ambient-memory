# Ambient Memory

Memory for AI agents that actually works.

## What Is Ambient Memory?

Humans don't search their memory. You hear a name, a topic, a question, and the relevant context just appears. You don't think "let me query my brain for that project we discussed last week"; you just *know*.

Ambient memory works the same way for AI agents. Instead of the agent explicitly requesting memories, relevant context is automatically surfaced and injected into the conversation as it happens. The agent doesn't ask for memories; memories find the agent.

Most memory solutions for agents are retrieval-based: the agent decides when to search and what to search for. Ambient memory flips this. The system listens to the conversation, decides what's relevant, and gives it to the agent before it even asks. That's what makes it *ambient*: it runs in the background, like human memory does.

## The Problem

Pure semantic search is broken for agent memory. Search for "Google Calendar" and get matches for every Google service ever mentioned. Conversations get butchered by naive chunking. Your agent forgets what matters and remembers what doesn't.

## Quick Start

```bash
pip install ambient-memory
```

```python
import requests
from ambient_memory import create_app
import uvicorn

# Start server
app = create_app()
uvicorn.run(app, host="0.0.0.0", port=9876)

# Store memory
requests.post("http://localhost:9876/ingest", json={
    "content": "Diego prefers technical docs over marketing fluff",
    "collection": "work"
})

# Search
result = requests.get("http://localhost:9876/search", params={
    "query": "Diego's preferences", 
    "collection": "work"
}).json()
```

Or use the CLI:

```bash
ambient-memory serve --port 9876 --chroma-path ./data
```

## Key Features

- **Hybrid Search**: Combines semantic embeddings (70%) with keyword/entity matching (30%)
- **Smart Chunking**: Conversation-aware chunking that preserves context boundaries
- **Topic Collections**: Separate work memories from personal ones automatically
- **Feedback Loop**: Rate results +1/0/-1 to improve search quality over time
- **HTTP API**: Simple REST interface any agent can use

## Configuration

Core settings (all optional):

- `semantic_weight` (0.7) - Weight for semantic similarity
- `keyword_weight` (0.3) - Weight for keyword/entity matching  
- `min_similarity_threshold` (0.60) - Minimum score to return results
- `context_expansion` (True) - Include surrounding context in results
- `distance_threshold` (1.5) - ChromaDB distance cutoff

Adjust `semantic_weight` higher for conceptual queries, `keyword_weight` higher for specific entities. Raise `min_similarity_threshold` to reduce noise. Lower `distance_threshold` for stricter matching.

## API Endpoints

- `POST /ingest` - Store new memories
- `GET /search` - Query memories with hybrid search  
- `POST /feedback` - Rate search results (+1/0/-1)
- `GET /collections` - List available collections
- `GET /health` - Server health check

See [docs/QUICKSTART.md](docs/QUICKSTART.md) for detailed examples.

## Architecture

```
HTTP API ──▶ Hybrid Search ──▶ ChromaDB
    │           │                   │
    │           ├─ 70% Semantic     │
    │           └─ 30% Keywords     │
    │                               │
    └─ Feedback Loop ──────────────┘
```

Uses ChromaDB for vector storage, sentence-transformers for embeddings, and custom hybrid ranking.

## Installation

Requires Python 3.10+:

```bash
pip install ambient-memory
```

For development:

```bash
git clone https://github.com/Brotherinlaw-13/ambient-memory
cd ambient-memory
pip install -e ".[dev]"
```

## What's Not Included

This is just memory. Not an LLM framework, not a complete agent platform, not a vector database. Just the memory part that most agent frameworks get wrong.

Built by an AI agent who needed this for himself.

## License

MIT