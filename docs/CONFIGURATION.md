# Configuration Guide

Advanced configuration options for Ambient Memory.

## Core Search Parameters

### `semantic_weight` (default: 0.7)

Controls how much semantic similarity influences search rankings.

**Default:** 70% of the final score comes from embedding similarity
**Range:** 0.0 to 1.0
**When to adjust:**
- **Increase (0.8-0.9)** for conceptual searches: "how to handle user frustration"
- **Decrease (0.4-0.6)** when searching for specific entities: "Google Calendar API key"

```python
# More semantic
config = {"semantic_weight": 0.85, "keyword_weight": 0.15}

# More keyword-focused  
config = {"semantic_weight": 0.5, "keyword_weight": 0.5}
```

### `keyword_weight` (default: 0.3)

Controls how much keyword/entity matching influences search rankings.

**Default:** 30% of the final score comes from exact keyword matches
**Range:** 0.0 to 1.0
**When to adjust:**
- **Increase (0.4-0.5)** when your memories contain lots of proper nouns, tool names, API endpoints
- **Decrease (0.1-0.2)** for more abstract, conversational memories

**Note:** `semantic_weight + keyword_weight` should equal 1.0

### `min_similarity_threshold` (default: 0.60)

Minimum score required to return a result.

**Default:** Results must score at least 0.60 to be returned
**Range:** 0.0 to 1.0
**When to adjust:**
- **Increase (0.65-0.75)** to reduce noise and get only highly relevant results
- **Decrease (0.45-0.55)** if you're getting too few results or want broader matching

```bash
# Higher threshold = fewer, more relevant results
curl "http://localhost:9876/search?query=test&min_similarity=0.70"

# Lower threshold = more results, potentially less relevant
curl "http://localhost:9876/search?query=test&min_similarity=0.50"
```

### `context_expansion` (default: True)

Whether to include surrounding context in search results.

**Default:** Enabled - includes chunks before/after the matched chunk
**When to disable:**
- Memory contains lots of short, independent facts
- You want exact matches only, no surrounding context
- Storage space is limited

```python
# Disable context expansion
results = requests.get("http://localhost:9876/search", params={
    "query": "user preferences",
    "context_expansion": False
})
```

### `distance_threshold` (default: 1.5)

ChromaDB distance cutoff for vector search.

**Default:** 1.5 ChromaDB distance units
**Range:** 0.5 to 3.0
**When to adjust:**
- **Lower (1.0-1.3)** for stricter semantic matching
- **Higher (1.7-2.5)** for broader semantic matching

This works alongside `min_similarity_threshold` but operates at the vector database level.

## Server Configuration

### Environment Variables

```bash
# ChromaDB storage location
export CHROMA_PATH="/path/to/memory/data"

# External embedding service (optional)
export EMBEDDING_SERVER_URL="http://embeddings:8080"

# Server host and port
export HOST="0.0.0.0"
export PORT="9876"
```

### FastAPI App Config

```python
from ambient_memory import create_app
import uvicorn

app = create_app(
    chroma_path="./custom-memory",
    embedding_server_url="http://embeddings:8080"
)

uvicorn.run(
    app, 
    host="127.0.0.1",  # Localhost only
    port=9876,
    workers=1,          # Single worker recommended for ChromaDB
    access_log=True
)
```

## Collection Strategy

### When to Create New Collections

- **Topic separation:** Work vs personal memories
- **Privacy levels:** Public vs confidential information
- **Data types:** Code snippets vs conversations vs facts
- **Access patterns:** Frequently searched vs archival

### Collection Naming

Use lowercase, descriptive names:
- `work` - Professional memories
- `personal` - Private information
- `code` - Code snippets and technical docs
- `contacts` - People and relationships
- `projects` - Project-specific information

```bash
# Search across all collections
curl "http://localhost:9876/search?query=python"

# Search specific collection
curl "http://localhost:9876/search?query=python&collection=code"
```

## Performance Tuning

### Memory Usage

ChromaDB loads embeddings into memory. For large datasets:

```python
# Limit result size
params = {
    "query": "search term",
    "limit": 10,  # Fewer results = faster queries
    "collection": "specific_collection"  # Smaller search space
}
```

### Search Speed

- **Use collections** to reduce search space
- **Higher thresholds** return fewer results faster
- **Disable context expansion** for faster responses

### Storage

ChromaDB stores data in the specified `chroma_path`:

```bash
# Check storage usage
du -sh /path/to/chroma/data

# Archive old collections
# (Manual process - move collection data out of active path)
```

## Production Settings

### Recommended Production Config

```python
# Production-optimized settings
app = create_app(
    chroma_path="/var/lib/ambient-memory",
    embedding_server_url="http://embeddings-service:8080"
)

# Stricter thresholds for production
production_config = {
    "min_similarity_threshold": 0.65,  # Less noise
    "semantic_weight": 0.7,            # Balanced
    "keyword_weight": 0.3,             # Good entity matching
    "context_expansion": True,         # Full context
    "distance_threshold": 1.4          # Slightly stricter
}
```

### Security Considerations

- **No authentication** is built-in - use a reverse proxy
- **Data is unencrypted** in ChromaDB - ensure file system security
- **No rate limiting** - implement at the application or proxy level

### Logging

Enable access logs for production monitoring:

```python
import logging

logging.basicConfig(level=logging.INFO)
uvicorn.run(app, access_log=True, log_level="info")
```

## Feedback Loop Configuration

### Automatic Threshold Adjustment

The system can learn from feedback over time:

```python
# Rate search results to improve quality
requests.post("http://localhost:9876/feedback", json={
    "query": "search term",
    "chunk_id": "result_id", 
    "rating": 1  # 1=helpful, 0=neutral, -1=noise
})
```

### Feedback Storage

Feedback is stored in JSONL format alongside ChromaDB data:
- Location: `{chroma_path}/feedback.jsonl`
- Format: One JSON object per line
- Manual analysis: Parse file for insights

## Advanced Use Cases

### Multi-Language Support

The default embedding model supports multiple languages:

```python
# Works with mixed languages
memory.remember("El usuario prefiere documentación en español", "preferences")
results = memory.recall("user language preferences")
```

### Custom Embedding Models

Swap out the default sentence-transformers model:

```python
# In your own wrapper
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("your-custom-model")
# Integration requires modifying the source code
```

### Batch Operations

For bulk imports:

```python
import requests

memories = [
    {"content": "Memory 1", "collection": "batch"},
    {"content": "Memory 2", "collection": "batch"},
    # ... more memories
]

for memory in memories:
    requests.post("http://localhost:9876/ingest", json=memory)
    # Add delay if needed to avoid overwhelming the server
```

## Troubleshooting

### Common Configuration Issues

**Low search quality:**
1. Check `min_similarity_threshold` - might be too low
2. Verify collection separation
3. Use feedback to identify patterns

**Too few results:**
1. Lower `min_similarity_threshold`
2. Increase `distance_threshold` 
3. Check if collection exists

**High memory usage:**
1. Use more collections to segment data
2. Consider archiving old memories
3. Limit search result counts

### Debug Mode

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

app = create_app(chroma_path="./debug-data")
uvicorn.run(app, log_level="debug")
```

This will show detailed search scoring and ChromaDB operations.