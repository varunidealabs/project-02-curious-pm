# Personal Memory Assistant - Callable Voice Assistant with RAG

A voice assistant you can call on a phone number that uses RAG memory to track your life and work history.

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

## 🚀 Production Deployment

1. **Get a domain**: Use services like Cloudflare, Namecheap
2. **Deploy backend**: Railway, Heroku, or DigitalOcean
3. **Update ElevenLabs tools**: Point to your production URL
4. **Test thoroughly**: Make test calls to verify memory storage/retrieval

## 🔐 Security Notes

- API uses Bearer token authentication
- All requests require valid `Authorization` header
- Store API keys securely in environment variables
- Use HTTPS in production

## 🎯 Next Steps

- Add more sophisticated entity extraction
- Implement user authentication for multi-user support
- Add memory categories and tagging
- Integrate calendar APIs for meeting reminders
- Add export functionality for stored memories