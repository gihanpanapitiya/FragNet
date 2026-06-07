#!/bin/bash
# Start the FragNet FastAPI backend
# Run from the project root: bash start_api.sh
source /Users/gihan/.env/fragnet/bin/activate
uvicorn fragnet.api.main:app --host 0.0.0.0 --port 8000 --reload
