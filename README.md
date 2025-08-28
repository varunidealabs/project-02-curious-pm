# Personal Memory Assistant - Callable Voice Assistant with RAG

A voice assistant you can call on a phone number that uses RAG memory to track your life and work history.

## Code Analysis - Personal Memory Assistant API

**Architecture Overview:**
This is a **vector-based memory storage and retrieval API** using RAG (Retrieval-Augmented Generation) architecture.

**Key Components:**
1. **FastAPI** - REST API framework
2. **Pinecone** - Vector database for semantic search
3. **Sentence Transformers** - Text embedding model
4. **Azure OpenAI** - Alternative embedding provider

**Application Flow:**

### 1. **Initialization (Lines 17-44)**
```
Environment Setup → Pinecone Connection → Index Creation → Embedding Model Load
```

### 2. **Store Memory Flow (/api/store-memory)**
```
Request → Auth Check → Generate UUID → Create Embedding → Store in Pinecone → Return Success
```
- Takes text content and metadata
- Converts to vector using `all-MiniLM-L6-v2` model
- Stores in Pinecone with metadata (type, entities, priority, timestamp)

### 3. **Search Memory Flow (/api/search-memory)**
```
Request → Auth Check → Query Embedding → Pinecone Search → Filter Results → Return Matches
```
- Converts search query to vector
- Performs semantic similarity search
- Filters by relevance score (>0.7) and metadata
- Returns top 5 relevant memories

### 4. **Authentication**
- Bearer token authentication using `API_SECRET_KEY`
- Required for all memory operations

### 5. **Health Check**
- Tests Pinecone connectivity
- Returns vector count and status

**Data Model:**
- **Memory Storage**: content, type, entities, priority, timestamp
- **Vector Dimension**: 384 (sentence-transformers model)
- **Search**: Cosine similarity with metadata filtering

## 🏗️ Architecture

```
Phone Call → ElevenLabs Agent → Custom Tools → Python API → Pinecone Vector DB
```

## 📋 Setup Instructions

### 1. Environment Setup

1. Copy the `.env` file and fill in your credentials:
```bash
cp .env.example .env
```

Required credentials:
- `PINECONE_API_KEY`: Get from https://pinecone.io
- `PINECONE_ENVIRONMENT`: Your Pinecone environment (e.g., "us-east1-gcp")
- `OPENAI_API_KEY`: For embeddings (optional, using sentence-transformers by default)
- `API_SECRET_KEY`: Generate a secure random string for API authentication

### 2. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the Server

```bash
python deploy.py
```

Or manually:
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Deploy to Production

For production deployment, use a service like:
- **Railway**: Connect GitHub repo, auto-deploy
- **Heroku**: `git push heroku main`
- **DigitalOcean App Platform**: Connect GitHub repo
- **AWS Lambda**: Use Mangum adapter

## 🔧 ElevenLabs Configuration

### Agent Settings
1. Go to https://elevenlabs.io/app/conversational-ai/agents
2. Create new agent with the system prompt from setup instructions
3. Configure phone number in Telephony settings

### Custom Tools Configuration
Add these two server tools to your ElevenLabs agent:

**Tool 1: store_memory**
- URL: `https://your-domain.com/api/store-memory`
- Method: POST
- Headers: `Authorization: Bearer YOUR_API_KEY`

**Tool 2: search_memory** 
- URL: `https://your-domain.com/api/search-memory`
- Method: POST
- Headers: `Authorization: Bearer YOUR_API_KEY`

## 🧪 Testing

### API Health Check
```bash
curl http://localhost:8000/api/health
```

### Test Memory Storage
```bash
curl -X POST "http://localhost:8000/api/store-memory" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Meeting with John tomorrow at 3 PM about marketing campaign",
    "memory_type": "meeting",
    "entities": "John, tomorrow 3PM, marketing campaign"
  }'
```

### Test Memory Search
```bash
curl -X POST "http://localhost:8000/api/search-memory" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was my meeting with John about?"
  }'
```

## 📞 Usage Flow

1. **Call the phone number** configured in ElevenLabs
2. **Speak naturally**: "I have a meeting with Sarah tomorrow at 2 PM about the budget review"
3. **Assistant responds**: "Got it! I've saved that you have a meeting with Sarah tomorrow at 2 PM about the budget review."
4. **Later, ask**: "What was my meeting with Sarah about?"
5. **Assistant retrieves**: "You had a meeting with Sarah about the budget review."

## 🔍 API Endpoints

- `GET /`: Root endpoint
- `POST /api/store-memory`: Store new memories
- `POST /api/search-memory`: Search stored memories  
- `GET /api/health`: Health check and stats
- `GET /docs`: Interactive API documentation
