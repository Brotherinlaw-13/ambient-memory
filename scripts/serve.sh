#!/bin/bash
cd /Users/rook/workspace/ambient-memory
.venv/bin/python3.12 -m uvicorn ambient_memory.server:app --host 0.0.0.0 --port 9877