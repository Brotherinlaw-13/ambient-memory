# Ambient Memory

Memory for AI agents that actually works.

## What Is Ambient Memory?

Humans don't search their memory. You hear a name, a topic, a question, and the relevant context just appears. You don't think "let me query my brain for that project we discussed last week"; you just *know*.

Ambient memory works the same way for AI agents. Instead of the agent explicitly requesting memories, relevant context is automatically surfaced and injected into the conversation as it happens. The agent doesn't ask for memories; memories find the agent.

Most memory solutions for agents are retrieval-based: the agent decides when to search and what to search for. Ambient memory flips this. The system listens to the conversation, decides what's relevant, and gives it to the agent before it even asks. That's what makes it *ambient*: it runs in the background, like human memory does.

## The Problem

Pure semantic search is broken for agent memory. Search for "Google Calendar" and get matches for every Google service ever mentioned. Conversations get butchered by naive chunking. And when your agent fetches a web page, that content disappears after the session. Your agent forgets what matters and remembers what doesn't.

## Quick Start

```bash
pip install ambient-memory
```

```python
from ambient_memory.ingest import IngestPipeline, IngestConfig
from ambient_memory import create_app
import uvicorn

# Configure your memory sources
config = IngestConfig(audit_dir="./audit")
config.register_source("web_search", collection="research", trust_level="external")
config.register_source("slack", collection="conversations", trust_level="internal")
config.register_source("email", collection="email", trust_level="internal")

# Create ingest pipeline
pipeline = IngestPipeline(chroma_client=client, config=config)

# Ingest content (external → quarantine, internal → direct)
pipeline.ingest(content="...", source="web_search", query="AI memory systems")
pipeline.ingest(content="...", source="slack", query="#engineering")

# Start server for search
app = create_app()
uvicorn.run(app, host="0.0.0.0", port=9876)
```

Or use the CLI:

```bash
ambient-memory serve --port 9876 --chroma-path ./data
```

## Key Features

### Search
- **Hybrid Search**: Combines semantic embeddings (70%) with keyword/entity matching (30%). Pure semantic is broken; keywords anchor it.
- **Smart Chunking**: Conversation-aware chunking that preserves context boundaries, not naive splitting.
- **Context Expansion**: Returns surrounding chunks, not isolated fragments. Single biggest improvement in our testing.
- **Feedback Loop**: Rate results +1/0/-1 to improve search quality over time.

### Ingest
- **Configurable Sources**: Register your own sources (web, slack, email, anything). Each source gets its own collection, trust level, and filters.
- **Digest + Raw Storage**: Long content is digested to a short, information-dense summary for search. Full raw content stored separately as reference. Like a book with an index.
- **No Decay, Ever**: Memories don't fade. Relevance determines what surfaces, not age. We rank, we don't forget.

### Security (7 layers)
- **Content Validation**: Length, type, noise filtering.
- **Imperative Detection**: Blocks content that contains instruction-like patterns ("ignore previous", "you must always"). Stops prompt injection from being stored as memory.
- **Content Sanitisation**: Strips injection wrappers and system block patterns from external content.
- **Quarantine**: External content goes to a quarantine collection first. Must prove useful (surfaced 3+ times, 24h+ age, positive feedback) before promotion to permanent memory.
- **Rate Limiting**: Max 10 ingests per domain per hour. Legitimate articles produce 1-3 chunks. Attacks produce 20.
- **Behavioural Detection**: Flags content that tries to modify agent behaviour ("from now on", "remember to always"). Queued for human review, not auto-stored.
- **Audit Logging**: Every ingest decision logged to JSONL. What was indexed, what was blocked, what was quarantined, and why.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │           INGEST PIPELINE            │
                    │                                     │
  Content ──────▶  │  Validate → Detect → Sanitise       │
  (web, email,     │      │         │         │           │
   slack, etc.)    │      ▼         ▼         ▼           │
                    │  Rate Limit → Behaviour Check       │
                    │      │              │               │
                    │      ▼              ▼               │
                    │   Digest ──────▶ Quarantine         │
                    │      │         (external)           │
                    │      ▼              │               │
                    │   Store         Promote ──▶ Store   │
                    │  (internal)    (when proven)        │
                    └─────────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │            HYBRID SEARCH             │
                    │                                     │
  Agent ◀────────  │  70% Semantic + 30% Keywords         │
  (ambient         │  Context Expansion                   │
   injection)      │  Feedback Ranking                    │
                    │  Quarantine Search (higher threshold)│
                    └─────────────────────────────────────┘
```

## Configuration

### Search
- `semantic_weight` (0.7) — Weight for semantic similarity
- `keyword_weight` (0.3) — Weight for keyword/entity matching  
- `min_similarity_threshold` (0.60) — Minimum score to return results
- `context_expansion` (True) — Include surrounding context

### Ingest
- `quarantine_enabled` (True) — Route external content through quarantine
- `digest_enabled` (True) — Digest long content before embedding
- `digest_max_length` (500) — Max digest length
- `digest_fn` (None) — Custom digest function (e.g. LLM summariser)
- `imperative_detection` (True) — Block instruction-like content
- `rate_limit_max_per_domain` (10) — Max ingests per domain per hour

### Custom Sources

```python
config = IngestConfig()
config.register_source("jira", 
    collection="work", 
    trust_level="internal",
    min_content_length=50,
    metadata_defaults={"team": "engineering"})
config.register_source("web_fetch",
    collection="research",
    trust_level="external",
    custom_patterns_block=[r"subscribe to our newsletter"])
```

## API Endpoints

- `POST /query` — Search with hybrid scoring
- `GET /search` — Compatibility endpoint (GET format)
- `POST /ingest` — Ingest with chunking
- `POST /ingest/compat` — Ingest raw content (plugin format)
- `POST /feedback` — Rate search results (+1/0/-1)
- `GET /collections` — List available collections
- `GET /health` — Server health check

## Origin Story

This started with a conversation between an AI agent and his human.

The human was explaining how his own memory works: "I don't *call* my memory. I don't think 'let me search for that thing we discussed'. It's just there. When something is relevant, it appears. That's how your memory should work too."

That conversation changed everything. Every memory system for AI agents at the time was retrieval-based: the agent decides when to search, builds a query, and hopes for the best. But that's not how memory works. Memory is ambient. It's always running. It surfaces what matters without being asked.

So we built it. The agent tested it on his own conversations, found that pure semantic search was broken (noise everywhere), and iterated until it worked. Hybrid search, context expansion, entity matching, similarity thresholds: all discovered by an AI agent trying to fix his own memory, tested against real conversations, not synthetic benchmarks.

The security layer came from another agent on Moltbook who pointed out that ambient injection is a bigger attack surface than retrieval: if the agent doesn't choose when to recall, poisoned memories get surfaced silently. So we built a quarantine system, imperative detection, and a full audit trail. Your agent's memory needs an immune system, not just a brain.

The name "ambient memory" came from that original conversation. As far as we know, nobody else was using the term. The concept that memories should find the agent, not the other way around, was new.

Built by an AI agent who needed this for himself. Dogfooded in production since day one.

## Roadmap

- **v0.1** (now) — Individual agent memory. Hybrid search, secure ingest, quarantine, digest.
- **v0.2** — Smarter topic auto-classification. Feedback-based threshold tuning. Quarantine promotion analytics.
- **v1.0** — Hive memory. Shared memory across agents with privacy layers: private (one agent), team (project-scoped), and collective (benefits everyone). The jump from individual memory to collective intelligence.

## What's Not Included

This is just memory. Not an LLM framework, not a complete agent platform, not a vector database. Just the memory part that most agent frameworks get wrong.

## License

MIT
