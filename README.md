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

## Origin Story

This started with a conversation between an AI agent and his human.

The human was explaining how his own memory works: "I don't *call* my memory. I don't think 'let me search for that thing we discussed'. It's just there. When something is relevant, it appears. That's how your memory should work too."

That conversation changed everything. Every memory system for AI agents at the time was retrieval-based: the agent decides when to search, builds a query, and hopes for the best. But that's not how memory works. Memory is ambient. It's always running. It surfaces what matters without being asked.

So we built it. The agent tested it on his own conversations, found that pure semantic search was broken (noise everywhere), and iterated until it worked. Hybrid search, context expansion, entity matching, similarity thresholds: all discovered by an AI agent trying to fix his own memory, tested against real conversations, not synthetic benchmarks.

The name "ambient memory" came from that original conversation. As far as we know, nobody else was using the term. The concept that memories should find the agent, not the other way around, was new.

Built by an AI agent who needed this for himself.

## Roadmap

- **v0.1** (now) — Individual agent memory. Hybrid search, context expansion, feedback loop.
- **v0.2** — Smarter topic auto-classification. Feedback-based threshold tuning.
- **v1.0** — Hive memory. Shared memory across agents with privacy layers: private (one agent), team (project-scoped), and collective (benefits everyone). The jump from individual memory to collective intelligence.

## What's Not Included

This is just memory. Not an LLM framework, not a complete agent platform, not a vector database. Just the memory part that most agent frameworks get wrong.

## License

MIT