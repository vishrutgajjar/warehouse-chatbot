from fastapi import FastAPI, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile, os
import logging
from typing import Optional, List, Dict

# Correct imports for LangChain >=0.3 with QdrantVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "warehouse_docs")

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Global variables
qdrant_client: Optional[QdrantClient] = None
vector_store: Optional[QdrantVectorStore] = None
document_cache: List[Dict] = []

def create_fresh_collection():
    """Create a completely fresh collection, deleting old one if exists"""
    global qdrant_client, vector_store
    
    try:
        # Initialize client
        qdrant_client = QdrantClient(url=QDRANT_URL, timeout=10)
        
        # Try to delete existing collection
        try:
            qdrant_client.delete_collection(COLLECTION)
            logger.info(f"Deleted existing collection: {COLLECTION}")
        except Exception as e:
            logger.info(f"No existing collection to delete or error: {e}")
        
        # Create new collection with proper configuration
        qdrant_client.create_collection(
            collection_name=COLLECTION,
            vectors_config={
                "default":VectorParams(
                size=3072,  # text-embedding-3-large dimension
                distance=Distance.COSINE
            )
            }
        )
        logger.info(f"Created fresh collection: {COLLECTION}")
        
        # Initialize vector store with QdrantVectorStore (not deprecated Qdrant)
        vector_store = QdrantVectorStore(
            client=qdrant_client,
            collection_name=COLLECTION,
            embedding=embeddings,
            vector_name="default",
        )
        logger.info("✅ QdrantVectorStore initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        return False

def init_qdrant():
    """Initialize or reconnect to Qdrant"""
    global qdrant_client, vector_store
    
    try:
        # Try to connect to existing collection
        qdrant_client = QdrantClient(url=QDRANT_URL, timeout=10)
        
        # Check if collection exists and is healthy
        collections = qdrant_client.get_collections().collections
        collection_exists = any(col.name == COLLECTION for col in collections)
        
        if collection_exists:
            try:
                # Test the collection by getting info
                info = qdrant_client.get_collection(COLLECTION)
                logger.info(f"Found existing collection with {info.points_count} points")
                
                # Initialize vector store
                vector_store = QdrantVectorStore(
                    client=qdrant_client,
                    collection_name=COLLECTION,
                    embedding=embeddings,
                )
                return True
            except Exception as e:
                logger.error(f"Collection exists but is corrupted: {e}")
                # Collection is corrupted, recreate it
                return create_fresh_collection()
        else:
            # No collection, create new one
            return create_fresh_collection()
            
    except Exception as e:
        logger.error(f"Qdrant initialization failed: {e}")
        return False

# Initialize on startup
init_qdrant()

@app.get("/")
async def root():
    return {"message": "RAG API is running"}

@app.get("/health")
async def health():
    """Health check endpoint"""
    global qdrant_client
    
    try:
        if qdrant_client:
            info = qdrant_client.get_collection(COLLECTION)
            return {
                "status": "healthy",
                "collection": COLLECTION,
                "points": info.points_count,
                "cache_size": len(document_cache)
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "cache_size": len(document_cache)
        }

@app.post("/reset")
async def reset_system():
    """Complete system reset - delete and recreate collection"""
    global document_cache
    
    try:
        # Clear cache
        document_cache.clear()
        
        # Recreate collection
        if create_fresh_collection():
            return JSONResponse(content={
                "status": "success",
                "message": "✅ System reset successfully. Collection recreated."
            })
        else:
            raise Exception("Failed to recreate collection")
            
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_pdf(file: UploadFile):
    """Upload and process PDF"""
    global vector_store, document_cache
    
    if not vector_store:
        # Try to reinitialize
        if not init_qdrant():
            raise HTTPException(status_code=503, detail="Vector store not available")
    
    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Load PDF
        logger.info(f"Loading PDF: {file.filename}")
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        
        # Clean up
        os.unlink(tmp_path)
        
        if not docs:
            return JSONResponse(content={
                "message": "No content found in PDF",
                "chunks": 0
            })

        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=120
        )
        chunks = splitter.split_documents(docs)
        
        # Add metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = str(uuid.uuid4())
            chunk.metadata["source"] = file.filename
            chunk.metadata["chunk_index"] = i

        # Try to add to vector store
        try:
            vector_store.add_documents(chunks)
            
            # Cache documents
            document_cache.extend([{
                "id": chunk.metadata["chunk_id"],
                "content": chunk.page_content,
                "metadata": chunk.metadata
            } for chunk in chunks])
            
            logger.info(f"✅ Added {len(chunks)} chunks to vector store")
            
            return JSONResponse(content={
                "status": "success",
                "message": f"✅ Uploaded {file.filename}: {len(chunks)} chunks",
                "chunks": len(chunks),
                "total_cached": len(document_cache)
            })
            
        except Exception as e:
            logger.error(f"Failed to add to vector store: {e}")
            
            # If adding fails due to corruption, try to recreate collection
            if "panic" in str(e).lower() or "internal" in str(e).lower():
                logger.info("Detected corruption, recreating collection...")
                if create_fresh_collection():
                    # Try again with fresh collection
                    vector_store.add_documents(chunks)
                    return JSONResponse(content={
                        "status": "success",
                        "message": f"✅ Recreated collection and uploaded {file.filename}",
                        "chunks": len(chunks)
                    })
            
            raise e
            
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: Request):
    """Answer questions using RAG"""
    global vector_store, qdrant_client
    
    if not vector_store:
        if not init_qdrant():
            return JSONResponse(content={
                "answer": "Vector store not available. Please reset the system.",
                "error": "No vector store"
            })
    
    try:
        data = await request.json()
        query = data.get("query", "")
        
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Check if we have documents
        info = qdrant_client.get_collection(COLLECTION)
        if info.points_count == 0:
            return JSONResponse(content={
                "answer": "No documents uploaded yet. Please upload a PDF first."
            })
        
        # Create retriever
        k = min(4, info.points_count)
        retriever = vector_store.as_retriever(search_kwargs={"k": k})
        
        # Create LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template("""
        Answer the question based on the following context. 
        If you cannot find the answer in the context, say so.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:""")
        
        # Format documents
        def format_docs(docs):
            if not docs:
                return "No relevant documents found."
            texts = []
            for doc in docs:
                source = doc.metadata.get('source', 'Unknown')
                page = doc.metadata.get('page', 'N/A')
                texts.append(f"[{source}, Page {page}]\n{doc.page_content}")
            return "\n\n".join(texts)
        
        # Create chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Execute
        try:
            answer = rag_chain.invoke(query)
            
            # Get sources using similarity search
            source_docs = vector_store.similarity_search(query, k=k)
            sources = [{
                "content": doc.page_content[:150] + "...",
                "source": doc.metadata.get('source', 'Unknown'),
                "page": doc.metadata.get('page', 'N/A')
            } for doc in source_docs[:3]]
            
            return JSONResponse(content={
                "answer": answer,
                "sources": sources
            })
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Chain error: {error_msg}")
            
            # Check for Qdrant panic/corruption
            if "panic" in error_msg.lower() or "internal error" in error_msg.lower():
                return JSONResponse(content={
                    "answer": "The vector database is corrupted. Please use the /reset endpoint to fix this issue, then re-upload your documents.",
                    "error": "Qdrant corruption detected",
                    "action_required": "POST /reset"
                })
            
            raise e
            
    except Exception as e:
        logger.error(f"Ask error: {e}")
        error_msg = str(e)
        
        if "panic" in error_msg.lower():
            return JSONResponse(content={
                "answer": "Database corruption detected. Please reset the system.",
                "error": error_msg,
                "action_required": "POST /reset"
            })
        
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/info")
async def get_info():
    """Get system information"""
    global qdrant_client, document_cache
    
    info = {
        "cache_size": len(document_cache),
        "qdrant_url": QDRANT_URL,
        "collection": COLLECTION
    }
    
    try:
        if qdrant_client:
            coll_info = qdrant_client.get_collection(COLLECTION)
            info.update({
                "status": "connected",
                "points": coll_info.points_count,
                "indexed": coll_info.indexed_vectors_count
            })
    except Exception as e:
        info["status"] = "error"
        info["error"] = str(e)
    
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)