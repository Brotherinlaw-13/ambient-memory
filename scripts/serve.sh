#!/bin/bash
# Serve ambient-memory on production port
# Use PORT env var to override (default: 9876)
cd /Users/rook/workspace/ambient-memory
PORT="${PORT:-9876}"
PYTHONPATH=src .venv/bin/python3.12 -m uvicorn ambient_memory.server:app --host 0.0.0.0 --port "$PORT"