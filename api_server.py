from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import uvicorn
from contextlib import asynccontextmanager

from rag_system import GovernmentJobRAG, RAGConfig, create_default_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG system instance
rag_system: Optional[GovernmentJobRAG] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global rag_system
    
    # Startup
    logger.info("Starting Government Job RAG API...")
    try:
        config = create_default_config()
        rag_system = GovernmentJobRAG(config)
        rag_system.initialize()
        logger.info("RAG system initialized successfully!")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Government Job RAG API...")

# Create FastAPI app
app = FastAPI(
    title="Government Job RAG API",
    description="A RAG (Retrieval-Augmented Generation) API for government job queries",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str = Field(..., description="The question to ask about government jobs")
    category: Optional[str] = Field(None, description="Filter by job category")
    education_level: Optional[str] = Field(None, description="Filter by education level (10th, 12th, graduate, postgraduate)")
    salary_range: Optional[List[int]] = Field(None, description="Salary range [min, max]")
    n_results: Optional[int] = Field(5, description="Number of results to return")

class UserProfile(BaseModel):
    education_level: Optional[str] = Field(None, description="User's education level")
    interests: Optional[List[str]] = Field(None, description="User's interests")
    experience_level: Optional[str] = Field(None, description="User's experience level")
    location: Optional[str] = Field(None, description="Preferred location")
    salary_expectations: Optional[str] = Field(None, description="Salary expectations")

class QueryResponse(BaseModel):
    answer: str
    relevant_jobs: List[Dict[str, Any]]
    sources: List[str]
    confidence_score: float
    query: str

class JobSearchRequest(BaseModel):
    query: str
    n_results: Optional[int] = Field(10, description="Number of results to return")

class JobSearchResponse(BaseModel):
    jobs: List[Dict[str, Any]]
    total_found: int
    query: str

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Government Job RAG API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "query": "/query",
            "recommendations": "/recommendations",
            "search": "/search",
            "categories": "/categories",
            "organizations": "/organizations",
            "stats": "/stats"
        }
    }

@app.post("/query", response_model=QueryResponse)
async def query_jobs(request: QueryRequest):
    """Query the RAG system with a question"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        # Extract salary range if provided
        salary_range = None
        if request.salary_range and len(request.salary_range) == 2:
            salary_range = (request.salary_range[0], request.salary_range[1])
        
        # Query the RAG system
        response = rag_system.query(
            question=request.question,
            category=request.category,
            education_level=request.education_level,
            salary_range=salary_range,
            n_results=request.n_results
        )
        
        return QueryResponse(
            answer=response.answer,
            relevant_jobs=response.relevant_jobs,
            sources=response.sources,
            confidence_score=response.confidence_score,
            query=request.question
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommendations", response_model=QueryResponse)
async def get_recommendations(profile: UserProfile):
    """Get personalized job recommendations"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        # Convert profile to dict
        profile_dict = profile.dict()
        
        # Get recommendations
        response = rag_system.get_job_recommendations(profile_dict)
        
        return QueryResponse(
            answer=response.answer,
            relevant_jobs=response.relevant_jobs,
            sources=response.sources,
            confidence_score=response.confidence_score,
            query="Job recommendations based on user profile"
        )
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=JobSearchResponse)
async def search_jobs(request: JobSearchRequest):
    """Search for jobs using vector similarity"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        # Search for jobs
        results = rag_system.vector_db.search_similar_jobs(
            query=request.query,
            n_results=request.n_results
        )
        
        return JobSearchResponse(
            jobs=results,
            total_found=len(results),
            query=request.query
        )
        
    except Exception as e:
        logger.error(f"Error searching jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/category/{category}")
async def search_by_category(category: str, n_results: int = Query(10, ge=1, le=50)):
    """Search jobs by category"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        results = rag_system.search_by_category(category, n_results)
        
        return {
            "category": category,
            "jobs": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching by category: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/education/{education}")
async def search_by_education(education: str, n_results: int = Query(10, ge=1, le=50)):
    """Search jobs by education level"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        results = rag_system.search_by_education(education, n_results)
        
        return {
            "education_level": education,
            "jobs": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching by education: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/organization/{organization}")
async def search_by_organization(organization: str, n_results: int = Query(10, ge=1, le=50)):
    """Search jobs by organization"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        results = rag_system.get_jobs_by_organization(organization)
        
        return {
            "organization": organization,
            "jobs": results[:n_results],
            "total_found": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching by organization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/categories")
async def get_categories():
    """Get all available job categories"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        categories = rag_system.get_all_categories()
        
        return {
            "categories": categories,
            "total_categories": len(categories)
        }
        
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/organizations")
async def get_organizations():
    """Get all organizations with jobs"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        # Get all jobs and extract unique organizations
        all_jobs = rag_system.processor.jobs
        organizations = list(set(job.organization for job in all_jobs))
        
        return {
            "organizations": sorted(organizations),
            "total_organizations": len(organizations)
        }
        
    except Exception as e:
        logger.error(f"Error getting organizations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    global rag_system
    
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        stats = rag_system.get_database_stats()
        
        # Add additional stats
        all_jobs = rag_system.processor.jobs
        categories = list(set(job.category for job in all_jobs))
        organizations = list(set(job.organization for job in all_jobs))
        
        stats.update({
            "total_categories": len(categories),
            "total_organizations": len(organizations),
            "categories": categories,
            "organizations": organizations[:10]  # Show first 10
        })
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global rag_system
    
    if not rag_system:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "RAG system not initialized"}
        )
    
    try:
        stats = rag_system.get_database_stats()
        return {
            "status": "healthy",
            "rag_system": "initialized",
            "total_jobs": stats.get("total_jobs", 0)
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": str(e)}
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "message": "The requested endpoint does not exist"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )



