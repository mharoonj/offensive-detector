#!/bin/bash

# FastAPI LLM Enhanced Toxic Classifier Server Startup Script

echo "🚀 Starting FastAPI LLM Enhanced Toxic Classifier Server"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run this script from the fastapi_classifier directory."
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import fastapi, uvicorn, transformers, torch, qdrant_client" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Error: Required packages not installed. Please run: pip install -r requirements.txt"
    exit 1
fi

# Check if Qdrant is running
echo "🔍 Checking Qdrant connection..."
python -c "
import qdrant_client
try:
    client = qdrant_client.QdrantClient(host='localhost', port=6333)
    collections = client.get_collections()
    print('✅ Qdrant is running')
    print(f'Available collections: {[col.name for col in collections.collections]}')
except Exception as e:
    print(f'❌ Qdrant connection failed: {e}')
    print('Please start Qdrant with: ./start_qdrant_simple.sh')
    exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(dirname $(pwd))"

echo ""
echo "🌐 Starting FastAPI server..."
echo "   API will be available at: http://localhost:8000"
echo "   Documentation at: http://localhost:8000/docs"
echo "   Press Ctrl+C to stop the server"
echo ""

# Start the server
python main.py 