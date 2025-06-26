#!/bin/bash

# Simple Qdrant Persistent Storage Script

echo "🚀 Setting up Qdrant with Persistent Storage..."

# Create directory for persistent storage
echo "📁 Creating persistent storage directory..."
mkdir -p ~/qdrant_data

# Set permissions
chmod 755 ~/qdrant_data

# Stop existing container if running
echo "🛑 Stopping existing Qdrant container..."
sudo docker stop qdrant-persistent 2>/dev/null || true
sudo docker rm qdrant-persistent 2>/dev/null || true

# Start Qdrant with persistent storage
echo "🐳 Starting Qdrant with persistent storage..."
sudo docker run -d \
  --name qdrant-persistent \
  -p 6333:6333 \
  -v ~/qdrant_data:/qdrant/storage \
  qdrant/qdrant

# Wait for startup
echo "⏳ Waiting for Qdrant to start..."
sleep 5

# Check if running
if curl -f http://localhost:6333/health > /dev/null 2>&1; then
    echo "✅ Qdrant is running successfully!"
    echo "📊 Health check: http://localhost:6333/health"
    echo "🔗 API: http://localhost:6333"
    echo "💾 Data stored in: ~/qdrant_data"
    echo ""
    echo "Commands:"
    echo "  Stop: sudo docker stop qdrant-persistent"
    echo "  Start: sudo docker start qdrant-persistent"
    echo "  Logs: sudo docker logs qdrant-persistent"
    echo "  Remove: sudo docker rm qdrant-persistent"
else
    echo "❌ Qdrant failed to start. Check logs:"
    sudo docker logs qdrant-persistent
fi 