# Qdrant Persistent Storage Commands

## 🚀 **Start Qdrant with Persistent Storage**

### **Method 1: Simple Docker Run**
```bash
# Create storage directory
mkdir -p ~/qdrant_data

# Start Qdrant with volume
sudo docker run -d \
  --name qdrant-persistent \
  -p 6333:6333 \
  -v ~/qdrant_data:/qdrant/storage \
  qdrant/qdrant
```

### **Method 2: Using Script**
```bash
# Make script executable
chmod +x start_qdrant_simple.sh

# Run the script
./start_qdrant_simple.sh
```

### **Method 3: Using Docker Compose**
```bash
# Start with docker-compose
docker-compose up -d

# Stop with docker-compose
docker-compose down
```

## 🔧 **Manage Qdrant Container**

### **Stop Qdrant**
```bash
sudo docker stop qdrant-persistent
```

### **Start Qdrant (if already created)**
```bash
sudo docker start qdrant-persistent
```

### **Restart Qdrant**
```bash
sudo docker restart qdrant-persistent
```

### **View Logs**
```bash
sudo docker logs qdrant-persistent
sudo docker logs -f qdrant-persistent  # Follow logs
```

### **Remove Container (keeps data)**
```bash
sudo docker rm qdrant-persistent
```

### **Remove Container and Data**
```bash
sudo docker rm qdrant-persistent
rm -rf ~/qdrant_data
```

## 📊 **Check Status**

### **Check if Running**
```bash
sudo docker ps | grep qdrant
```

### **Health Check**
```bash
curl http://localhost:6333/health
```

### **Check Collections**
```bash
curl http://localhost:6333/collections
```

## 💾 **Data Location**

- **Local Directory**: `~/qdrant_data`
- **Container Path**: `/qdrant/storage`
- **Data Persistence**: ✅ Yes (survives container restart)

## 🔍 **Troubleshooting**

### **Permission Issues**
```bash
sudo chown -R $USER:$USER ~/qdrant_data
chmod 755 ~/qdrant_data
```

### **Port Already in Use**
```bash
# Check what's using port 6333
sudo lsof -i :6333

# Kill process if needed
sudo kill -9 <PID>
```

### **Container Won't Start**
```bash
# Check logs
sudo docker logs qdrant-persistent

# Check disk space
df -h

# Check Docker status
sudo systemctl status docker
```

## 📝 **Example Usage**

```bash
# 1. Start Qdrant with persistence
./start_qdrant_simple.sh

# 2. Load your data
python load_jigsaw_to_qdrant_enhanced.py

# 3. Stop Qdrant
sudo docker stop qdrant-persistent

# 4. Start again (data will be there)
sudo docker start qdrant-persistent

# 5. Check your data is still there
curl http://localhost:6333/collections
``` 