# Quick Start Guide

Step-by-step guide to get Ambient Memory running for your AI agent.

## 1. Installation

```bash
pip install ambient-memory
```

Requirements: Python 3.10+

## 2. Start the Server

### Option A: CLI (Recommended)

```bash
ambient-memory serve --port 9876 --chroma-path ./memory-data
```

### Option B: Python Code

```python
import uvicorn
from ambient_memory import create_app

app = create_app(chroma_path="./memory-data")
uvicorn.run(app, host="0.0.0.0", port=9876)
```

The server will start at `http://localhost:9876`. ChromaDB data will be stored in `./memory-data/`.

## 3. Store Memories

```bash
curl -X POST "http://localhost:9876/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Diego prefers technical documentation over marketing copy",
    "collection": "work"
  }'
```

Response:
```json
{
  "success": true,
  "chunk_id": "chunk_12345",
  "collection": "work"
}
```

## 4. Search Memories

```bash
curl "http://localhost:9876/search?query=Diego%20preferences&collection=work&limit=5"
```

Response:
```json
{
  "results": [
    {
      "content": "Diego prefers technical documentation over marketing copy",
      "similarity_score": 0.87,
      "chunk_id": "chunk_12345",
      "collection": "work"
    }
  ],
  "query": "Diego preferences",
  "total_results": 1
}
```

## 5. Give Feedback

Rate results to improve search quality over time:

```bash
curl -X POST "http://localhost:9876/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Diego preferences", 
    "chunk_id": "chunk_12345",
    "rating": 1
  }'
```

Ratings: `1` = helpful, `0` = neutral, `-1` = noise

## Working with Collections

Collections separate different types of memories. Common patterns:

- `work` - Professional memories
- `personal` - Private information  
- `technical` - Code snippets, documentation
- `contacts` - People and relationships

```bash
# List all collections
curl "http://localhost:9876/collections"

# Search specific collection
curl "http://localhost:9876/search?query=python%20code&collection=technical"

# Store in specific collection
curl -X POST "http://localhost:9876/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Use list comprehensions for filtering data in Python",
    "collection": "technical"
  }'
```

## Python Integration

```python
import requests

class AgentMemory:
    def __init__(self, base_url="http://localhost:9876"):
        self.base_url = base_url
    
    def remember(self, content, collection="general"):
        """Store a memory."""
        response = requests.post(f"{self.base_url}/ingest", json={
            "content": content,
            "collection": collection
        })
        return response.json()
    
    def recall(self, query, collection=None, limit=5):
        """Search memories."""
        params = {"query": query, "limit": limit}
        if collection:
            params["collection"] = collection
        
        response = requests.get(f"{self.base_url}/search", params=params)
        return response.json()["results"]
    
    def feedback(self, query, chunk_id, rating):
        """Rate a search result."""
        requests.post(f"{self.base_url}/feedback", json={
            "query": query,
            "chunk_id": chunk_id, 
            "rating": rating
        })

# Usage
memory = AgentMemory()
memory.remember("User prefers JSON over XML for API responses", "preferences")
results = memory.recall("API format preferences")
if results:
    memory.feedback("API format preferences", results[0]["chunk_id"], 1)
```

## Configuration

Server accepts these environment variables:

- `CHROMA_PATH` - ChromaDB storage directory (default: `.chromadb`)
- `EMBEDDING_SERVER_URL` - External embedding service URL (optional)

Or configure via create_app():

```python
from ambient_memory import create_app

app = create_app(
    chroma_path="./custom-memory-path",
    embedding_server_url="http://embeddings:8080"
)
```

## Health Check

```bash
curl "http://localhost:9876/health"
```

Returns `{"status": "healthy"}` if the server is running and ChromaDB is accessible.

## Troubleshooting

**Server won't start:**
- Check if port 9876 is available: `lsof -i :9876`
- Ensure Python 3.10+ is installed
- Check ChromaDB permissions on data directory

**No search results:**
- Verify collection name matches
- Try searching without collection filter
- Check similarity threshold in configuration

**Poor search quality:**
- Use feedback ratings to train the system
- Separate different topics into collections
- Consider adjusting semantic/keyword weights

## Next Steps

- See [CONFIGURATION.md](CONFIGURATION.md) for advanced settings
- Check the [GitHub repo](https://github.com/Brotherinlaw-13/ambient-memory) for examples
- Join discussions in GitHub Issues