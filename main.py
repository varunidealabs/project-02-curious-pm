from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from typing import Optional, List
from pinecone import Pinecone, ServerlessSpec
from openai import AzureOpenAI
import os
from datetime import datetime 
import uuid
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Personal Memory Assistant API")

# Debug storage
debug_logs = []

def add_debug_log(step: str, status: str, details: str = ""):
    """Add debug log entry"""
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "step": step,
        "status": status,
        "details": details
    }
    debug_logs.append(log_entry)
    # Keep only last 100 logs
    if len(debug_logs) > 100:
        debug_logs.pop(0)
    print(f"[DEBUG] {step}: {status} - {details}")

# Initialize Pinecone
add_debug_log("STARTUP", "INITIALIZING", "Starting Pinecone initialization")
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    add_debug_log("PINECONE_INIT", "SUCCESS", "Pinecone client initialized")
except Exception as e:
    add_debug_log("PINECONE_INIT", "FAILED", str(e))
    raise e

# Create or connect to Pinecone index
INDEX_NAME = "personal-memory"
DIMENSION = 1536  # text-embedding-ada-002 dimension

add_debug_log("INDEX_CHECK", "STARTING", f"Checking if index '{INDEX_NAME}' exists")
try:
    existing_indexes = [index.name for index in pc.list_indexes()]
    add_debug_log("INDEX_LIST", "SUCCESS", f"Found {len(existing_indexes)} indexes")
    
    if INDEX_NAME not in existing_indexes:
        add_debug_log("INDEX_CREATE", "STARTING", f"Creating index '{INDEX_NAME}'")
        pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        add_debug_log("INDEX_CREATE", "SUCCESS", f"Index '{INDEX_NAME}' created")
    else:
        add_debug_log("INDEX_EXISTS", "SUCCESS", f"Index '{INDEX_NAME}' already exists")
    
    index = pc.Index(INDEX_NAME)
    add_debug_log("INDEX_CONNECT", "SUCCESS", f"Connected to index '{INDEX_NAME}'")
except Exception as e:
    add_debug_log("INDEX_SETUP", "FAILED", str(e))
    raise e

# Initialize Azure OpenAI client
add_debug_log("AZURE_INIT", "STARTING", "Initializing Azure OpenAI client")
try:
    azure_client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )
    add_debug_log("AZURE_INIT", "SUCCESS", "Azure OpenAI client initialized")
except Exception as e:
    add_debug_log("AZURE_INIT", "FAILED", str(e))
    azure_client = None

def get_embedding(text: str, use_azure: bool = True):
    """Get embedding for text using Azure OpenAI"""
    add_debug_log("EMBEDDING_START", "STARTING", f"Creating embedding for text length: {len(text)}")
    
    if use_azure and azure_client:
        try:
            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT", "text-embedding-ada-002")
            add_debug_log("EMBEDDING_REQUEST", "STARTING", f"Using deployment: {deployment_name}")
            
            response = azure_client.embeddings.create(
                model=deployment_name,
                input=text
            )
            add_debug_log("EMBEDDING_SUCCESS", "SUCCESS", f"Embedding created, dimension: {len(response.data[0].embedding)}")
            return response.data[0].embedding
        except Exception as e:
            add_debug_log("EMBEDDING_FAILED", "FAILED", str(e))
            raise HTTPException(status_code=500, detail=f"Azure embedding failed: {str(e)}")
    else:
        add_debug_log("EMBEDDING_NO_CLIENT", "FAILED", "Azure OpenAI client not available")
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
    add_debug_log("STORE_MEMORY", "STARTING", f"Storing memory of type: {request.memory_type}")
    
    try:
        # Generate unique ID for memory
        memory_id = str(uuid.uuid4())
        add_debug_log("MEMORY_ID", "SUCCESS", f"Generated ID: {memory_id}")
        
        # Create embedding for the content
        add_debug_log("EMBEDDING_CALL", "STARTING", "Calling get_embedding function")
        embedding = get_embedding(request.content)
        add_debug_log("EMBEDDING_RECEIVED", "SUCCESS", f"Received embedding with {len(embedding)} dimensions")
        
        # Prepare metadata
        metadata = {
            "content": request.content,
            "memory_type": request.memory_type,
            "entities": request.entities or "",
            "priority": request.priority,
            "timestamp": datetime.utcnow().isoformat(),
            "date_created": datetime.utcnow().strftime("%Y-%m-%d"),
        }
        add_debug_log("METADATA_PREPARED", "SUCCESS", f"Metadata prepared for {request.memory_type}")
        
        # Store in Pinecone
        add_debug_log("PINECONE_UPSERT", "STARTING", f"Upserting vector to Pinecone")
        index.upsert([
            {
                "id": memory_id,
                "values": embedding,
                "metadata": metadata
            }
        ])
        add_debug_log("PINECONE_UPSERT", "SUCCESS", f"Vector stored successfully")
        
        add_debug_log("STORE_MEMORY", "SUCCESS", f"Memory {memory_id} stored successfully")
        return StoreMemoryResponse(
            success=True,
            memory_id=memory_id,
            message="Memory stored successfully"
        )
        
    except Exception as e:
        add_debug_log("STORE_MEMORY", "FAILED", str(e))
        raise HTTPException(status_code=500, detail=f"Error storing memory: {str(e)}")

@app.post("/api/search-memory", response_model=SearchMemoryResponse)
def search_memory(request: SearchMemoryRequest, _: str = Depends(verify_api_key)):
    add_debug_log("SEARCH_MEMORY", "STARTING", f"Searching for: {request.query}")
    
    try:
        # Create embedding for search query
        add_debug_log("SEARCH_EMBEDDING", "STARTING", "Creating embedding for search query")
        query_embedding = get_embedding(request.query)
        add_debug_log("SEARCH_EMBEDDING", "SUCCESS", "Query embedding created")
        
        # Build filter
        filter_dict = {}
        if request.memory_type:
            filter_dict["memory_type"] = {"$eq": request.memory_type}
            add_debug_log("FILTER_TYPE", "SUCCESS", f"Added memory_type filter: {request.memory_type}")
            
        # Time-based filtering
        if request.time_range:
            today = datetime.utcnow().strftime("%Y-%m-%d")
            if "today" in request.time_range.lower():
                filter_dict["date_created"] = {"$eq": today}
                add_debug_log("FILTER_TIME", "SUCCESS", f"Added date filter: {today}")
        
        add_debug_log("PINECONE_QUERY", "STARTING", f"Querying Pinecone with filters: {filter_dict}")
        # Search in Pinecone
        search_results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        add_debug_log("PINECONE_QUERY", "SUCCESS", f"Found {len(search_results['matches'])} raw matches")
        
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
        
        add_debug_log("SEARCH_MEMORY", "SUCCESS", f"Returning {len(memories)} memories above 0.7 threshold")
        return SearchMemoryResponse(
            success=True,
            memories=memories,
            count=len(memories)
        )
        
    except Exception as e:
        add_debug_log("SEARCH_MEMORY", "FAILED", str(e))
        raise HTTPException(status_code=500, detail=f"Error searching memories: {str(e)}")

@app.get("/api/health")
def health_check():
    add_debug_log("HEALTH_CHECK", "STARTING", "Checking system health")
    try:
        # Test Pinecone connection
        add_debug_log("HEALTH_PINECONE", "STARTING", "Testing Pinecone connection")
        stats = index.describe_index_stats()
        add_debug_log("HEALTH_PINECONE", "SUCCESS", f"Pinecone connected, {stats['total_vector_count']} vectors")
        
        add_debug_log("HEALTH_CHECK", "SUCCESS", "All systems healthy")
        return {
            "status": "healthy",
            "pinecone_connected": True,
            "azure_openai_connected": azure_client is not None,
            "total_vectors": stats["total_vector_count"],
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        add_debug_log("HEALTH_CHECK", "FAILED", str(e))
        return {
            "status": "unhealthy", 
            "error": str(e),
            "pinecone_connected": False,
            "azure_openai_connected": azure_client is not None,
            "timestamp": datetime.utcnow().isoformat()
        }

@app.get("/api/debug")
def get_debug_logs():
    """Get debug logs for troubleshooting"""
    return {
        "total_logs": len(debug_logs),
        "latest_logs": debug_logs[-20:],  # Last 20 logs
        "system_status": {
            "pinecone_initialized": 'pc' in globals(),
            "azure_client_available": azure_client is not None,
            "index_name": INDEX_NAME,
            "dimension": DIMENSION
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
