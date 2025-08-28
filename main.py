from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional, List
import pinecone
from openai import AzureOpenAI
# from sentence_transformers import SentenceTransformer  # Removed to reduce image size
import os
from datetime import datetime
import json
import uuid
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Personal Memory Assistant API")

# Initialize Pinecone (old way for v2.2.4)
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment="us-east-1")

# Create or connect to Pinecone index
INDEX_NAME = "personal-memory"
DIMENSION = 1536  # text-embedding-ada-002 dimension

# Check if index exists, create if not
existing_indexes = pinecone.list_indexes()
if INDEX_NAME not in existing_indexes:
    pinecone.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine"
    )

index = pinecone.Index(INDEX_NAME)

# Initialize Azure OpenAI client (optional for better embeddings)
print(f"[DEBUG] Initializing Azure OpenAI...")
print(f"[DEBUG] API Key present: {bool(os.getenv('AZURE_OPENAI_API_KEY'))}")
print(f"[DEBUG] Endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
print(f"[DEBUG] API Version: {os.getenv('AZURE_OPENAI_API_VERSION')}")
print(f"[DEBUG] Deployment: {os.getenv('AZURE_OPENAI_DEPLOYMENT')}")

try:
    azure_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    print(f"[DEBUG] Azure OpenAI client initialized successfully")
except Exception as e:
    print(f"[DEBUG] Azure OpenAI initialization failed: {e}")
    print(f"[DEBUG] Error type: {type(e)}")
    azure_client = None

# Initialize embedding model (using Azure OpenAI as primary)
# embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Commented out to reduce image size
embedder = None  # Will use Azure OpenAI only

def get_embedding(text: str, use_azure: bool = True):
    """Get embedding for text using Azure OpenAI (sentence-transformers removed for smaller image)"""
    print(f"[DEBUG] get_embedding called with use_azure={use_azure}, azure_client={azure_client is not None}")
    
    if use_azure and azure_client:
        try:
            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "text-embedding-ada-002")
            print(f"[DEBUG] Using deployment: {deployment_name}")
            print(f"[DEBUG] Azure endpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
            print(f"[DEBUG] API version: {os.getenv('AZURE_OPENAI_API_VERSION')}")
            
            response = azure_client.embeddings.create(
                model=deployment_name,
                input=text
            )
            print(f"[DEBUG] Azure OpenAI response received successfully")
            return response.data[0].embedding
        except Exception as e:
            print(f"[DEBUG] Azure embedding error: {str(e)}")
            print(f"[DEBUG] Error type: {type(e)}")
            raise HTTPException(status_code=500, detail=f"Azure embedding failed: {str(e)}")
    else:
        print(f"[DEBUG] Azure OpenAI not available - azure_client: {azure_client}")
        raise HTTPException(status_code=500, detail="Azure OpenAI is required but not configured")

# Pydantic models
class StoreMemoryRequest(BaseModel):
    content: str
    memory_type: str
    entities: Optional[str] = None
    priority: Optional[str] = "medium"

class SearchMemoryRequest(BaseModel):
    query: str
    memory_type: Optional[str] = None
    time_range: Optional[str] = None

class StoreMemoryResponse(BaseModel):
    success: bool
    memory_id: str
    message: str

class SearchMemoryResponse(BaseModel):
    success: bool
    memories: List[dict]
    count: int

# Authentication dependency
def verify_api_key(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        
        if token != os.getenv("API_SECRET_KEY"):
            raise HTTPException(status_code=401, detail="Invalid API key")
            
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

@app.get("/")
def root():
    return {"message": "Personal Memory Assistant API", "status": "running"}

@app.post("/api/store-memory", response_model=StoreMemoryResponse)
def store_memory(request: StoreMemoryRequest, _: str = Depends(verify_api_key)):
    try:
        print(f"[DEBUG] Starting store_memory with content: {request.content[:50]}...")
        
        # Generate unique ID for memory
        memory_id = str(uuid.uuid4())
        print(f"[DEBUG] Generated memory_id: {memory_id}")
        
        # Create embedding for the content
        print(f"[DEBUG] Creating embedding...")
        embedding = get_embedding(request.content)
        print(f"[DEBUG] Embedding created successfully, length: {len(embedding)}")
        
        # Prepare metadata
        metadata = {
            "content": request.content,
            "memory_type": request.memory_type,
            "entities": request.entities or "",
            "priority": request.priority,
            "timestamp": datetime.utcnow().isoformat(),
            "date_created": datetime.utcnow().strftime("%Y-%m-%d"),
        }
        print(f"[DEBUG] Metadata prepared: {metadata}")
        
        # Store in Pinecone
        print(f"[DEBUG] Upserting to Pinecone...")
        index.upsert([
            {
                "id": memory_id,
                "values": embedding,
                "metadata": metadata
            }
        ])
        print(f"[DEBUG] Successfully stored in Pinecone")
        
        return StoreMemoryResponse(
            success=True,
            memory_id=memory_id,
            message="Memory stored successfully"
        )
        
    except Exception as e:
        print(f"Store memory error: {str(e)}")  # Log to console
        print(f"Error type: {type(e)}")  # Log error type
        raise HTTPException(status_code=500, detail=f"Error storing memory: {str(e)}")

@app.post("/api/search-memory", response_model=SearchMemoryResponse)
def search_memory(request: SearchMemoryRequest, _: str = Depends(verify_api_key)):
    try:
        # Create embedding for search query
        query_embedding = get_embedding(request.query)
        
        # Build filter
        filter_dict = {}
        if request.memory_type:
            filter_dict["memory_type"] = {"$eq": request.memory_type}
            
        # Time-based filtering (simplified)
        if request.time_range:
            # This is a basic implementation - you can enhance with more sophisticated date parsing
            today = datetime.utcnow().strftime("%Y-%m-%d")
            if "today" in request.time_range.lower():
                filter_dict["date_created"] = {"$eq": today}
        
        # Search in Pinecone
        search_results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        # Format results
        memories = []
        for match in search_results["matches"]:
            if match["score"] > 0.7:  # Relevance threshold
                memory = {
                    "id": match["id"],
                    "content": match["metadata"]["content"],
                    "memory_type": match["metadata"]["memory_type"],
                    "entities": match["metadata"]["entities"],
                    "timestamp": match["metadata"]["timestamp"],
                    "relevance_score": match["score"]
                }
                memories.append(memory)
        
        return SearchMemoryResponse(
            success=True,
            memories=memories,
            count=len(memories)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching memories: {str(e)}")

@app.get("/api/health")
def health_check():
    try:
        # Test Pinecone connection
        stats = index.describe_index_stats()
        return {
            "status": "healthy",
            "pinecone_connected": True,
            "total_vectors": stats["total_vector_count"],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
     import uvicorn
     port = int(os.getenv("PORT", 8000))
     uvicorn.run(app, host="0.0.0.0", port=port)
