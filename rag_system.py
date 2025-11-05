import os
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
import re
from dataclasses import dataclass, asdict
import json
from pathlib import Path

from data_processor import JobDataProcessor, JobData
from vector_database import VectorDatabase, JobRetriever
from rag_generator import JobRAGGenerator, SimpleRAGGenerator, RAGResponse

logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_db_path: str = "./chroma_db"
    collection_name: str = "government_jobs"
    llm_model: str = "gpt-3.5-turbo"
    llm_temperature: float = 0.7
    max_tokens: int = 1000
    use_openai: bool = False
    openai_api_key: Optional[str] = None
    n_results: int = 5

class GovernmentJobRAG:
    """Complete RAG system for government job queries"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.processor = None
        self.vector_db = None
        self.retriever = None
        self.generator = None
        self.is_initialized = False
        
    def initialize(self, data_dir: str = "."):
        """Initialize the RAG system"""
        try:
            logger.info("Initializing Government Job RAG System...")
            
            # Initialize data processor
            self.processor = JobDataProcessor(data_dir)
            
            # Initialize vector database
            self.vector_db = VectorDatabase(
                embedding_model_name=self.config.embedding_model,
                persist_directory=self.config.vector_db_path,
                collection_name=self.config.collection_name
            )
            
            # Initialize retriever
            self.retriever = JobRetriever(self.vector_db)
            
            # Initialize generator
            if self.config.use_openai and self.config.openai_api_key:
                self.generator = JobRAGGenerator(
                    model_name=self.config.llm_model,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.max_tokens,
                    openai_api_key=self.config.openai_api_key
                )
            else:
                self.generator = SimpleRAGGenerator()
            
            # Load and index data
            self._load_and_index_data()
            
            self.is_initialized = True
            logger.info("RAG system initialized successfully!")
            
        except Exception as e:
            logger.error(f"Error initializing RAG system: {e}")
            raise
    
    def _load_and_index_data(self):
        """Load job data and create vector embeddings"""
        try:
            # Load jobs
            jobs = self.processor.load_all_jobs()
            
            if not jobs:
                logger.warning("No jobs loaded from data files")
                return
            
            # Check if data is already indexed
            stats = self.vector_db.get_database_stats()
            if stats.get('total_jobs', 0) > 0:
                logger.info(f"Database already contains {stats['total_jobs']} jobs")
                return
            
            # Add jobs to vector database
            logger.info(f"Indexing {len(jobs)} jobs...")
            self.vector_db.add_jobs_to_database(jobs, self.processor)
            
            logger.info("Data indexing completed!")
            
        except Exception as e:
            logger.error(f"Error loading and indexing data: {e}")
            raise
    
    def query(self, 
              question: str,
              category: Optional[str] = None,
              education_level: Optional[str] = None,
              salary_range: Optional[tuple] = None,
              n_results: Optional[int] = None) -> RAGResponse:
        """Query the RAG system"""
        
        if not self.is_initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")
        
        try:
            # Set number of results
            n_results = n_results or self.config.n_results
            
            # Extract filters from natural language (best-effort) and merge with explicit params
            extracted = extract_filters(question)
            # Resolve category against known categories for exact-match filtering in Chroma
            extracted_sector = extracted.get("sector")
            candidate_category = category or extracted_sector
            resolved_category = self._resolve_category(candidate_category) if candidate_category else None
            resolved_education = education_level or extracted.get("education_level")
            extracted_salary: Optional[Tuple[Optional[int], Optional[int]]] = None
            if extracted.get("salary_min") is not None or extracted.get("salary_max") is not None:
                extracted_salary = (extracted.get("salary_min"), extracted.get("salary_max"))
            resolved_salary = salary_range or extracted_salary
            
            # Retrieve relevant jobs
            relevant_jobs = self.retriever.find_jobs(
                query=question,
                category=resolved_category,
                education_level=resolved_education,
                salary_range=resolved_salary,
                n_results=n_results
            )
            
            # Generate response
            response = self.generator.generate_response(question, relevant_jobs)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return RAGResponse(
                answer=f"I apologize, but I encountered an error while processing your query: {str(e)}",
                relevant_jobs=[],
                sources=[],
                confidence_score=0.0
            )
    
    def get_job_recommendations(self, user_profile: Dict[str, Any]) -> RAGResponse:
        """Get personalized job recommendations"""
        
        if not self.is_initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")
        
        try:
            # Find relevant jobs based on profile
            relevant_jobs = self.retriever.get_job_recommendations(
                user_profile, 
                n_results=self.config.n_results
            )
            
            # Generate recommendations
            if isinstance(self.generator, JobRAGGenerator):
                response = self.generator.generate_job_recommendations(
                    user_profile, relevant_jobs
                )
            else:
                question = f"Recommend jobs for someone with {user_profile.get('education_level', 'unknown education')} and interests in {', '.join(user_profile.get('interests', []))}"
                response = self.generator.generate_response(question, relevant_jobs)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return RAGResponse(
                answer=f"I apologize, but I encountered an error while generating recommendations: {str(e)}",
                relevant_jobs=[],
                sources=[],
                confidence_score=0.0
            )
    
    def search_by_category(self, category: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search jobs by category"""
        if not self.is_initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")
        
        return self.vector_db.search_by_category(category, n_results)
    
    def search_by_education(self, education: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search jobs by education level"""
        if not self.is_initialized:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")
        
        return self.vector_db.search_by_education_level(education, n_results)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        if not self.is_initialized:
            return {"error": "RAG system not initialized"}
        
        return self.vector_db.get_database_stats()
    
    def get_all_categories(self) -> List[str]:
        """Get all available job categories"""
        if not self.is_initialized:
            return []
        
        try:
            # Get all jobs and extract unique categories
            all_jobs = self.processor.jobs
            categories = list(set(job.category for job in all_jobs))
            return sorted(categories)
        except Exception as e:
            logger.error(f"Error getting categories: {e}")
            return []
    
    def get_jobs_by_organization(self, organization: str) -> List[Dict[str, Any]]:
        """Get jobs by organization"""
        if not self.is_initialized:
            return []
        
        try:
            results = self.vector_db.search_similar_jobs(
                query=f"organization: {organization}",
                n_results=50
            )
            return results
        except Exception as e:
            logger.error(f"Error searching by organization: {e}")
            return []
    
    def export_results(self, results: List[Dict[str, Any]], filename: str):
        """Export search results to JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results exported to {filename}")
        except Exception as e:
            logger.error(f"Error exporting results: {e}")

    def _resolve_category(self, sector_or_category: Optional[str]) -> Optional[str]:
        """Map user-provided sector/category to an exact category present in data.
        Performs case-insensitive and prefix matching, plus a few common synonyms.
        """
        if not sector_or_category:
            return None
        user_val = sector_or_category.strip()
        if not user_val:
            return None
        synonyms = {
            "medical": "Healthcare / Medical",
            "health": "Healthcare / Medical",
            "healthcare": "Healthcare / Medical",
            "bank": "Banking",
            "banking": "Banking & Financial Institutions",
            "finance": "Banking & Financial Institutions",
            "defense": "Defence",
            "defence": "Defence",
            "railway": "Railways",
            "railways": "Railways",
            "research": "Research",
            "judiciary": "Judiciary",
            "admin": "Administration",
            "administration": "Administration",
            "psu": "PSU",
            "upsc": "UPSC",
            "ssc": "SSC",
        }
        key = user_val.lower()
        if key in synonyms:
            return synonyms[key]
        # derive available categories from loaded jobs
        try:
            categories = list(set(job.category for job in (self.processor.jobs or [])))
        except Exception:
            categories = []
        if not categories:
            return user_val
        # exact case-insensitive
        for c in categories:
            if c.lower() == key:
                return c
        # prefix/substring match
        for c in categories:
            if key in c.lower() or c.lower() in key:
                return c
        # fallback: do not force a category that doesn't exist; let semantic search work
        return None

def _parse_number_with_suffix(token: str) -> Optional[int]:
    """Parse numbers like '10k', '25K', '20000', '1.5lakh' to integer rupees."""
    try:
        t = token.strip().lower().replace(",", "")
        match = re.match(r"^(\d+(?:\.\d+)?)\s*(k|k\+|k\.|thousand|l|lakh|lac|lakhs|lacs)?$", t)
        if not match:
            # bare int
            if t.isdigit():
                return int(t)
            return None
        value = float(match.group(1))
        unit = match.group(2) or ""
        if unit.startswith("k") or unit == "thousand":
            return int(value * 1000)
        if unit in {"l", "lakh", "lac", "lakhs", "lacs"}:
            return int(value * 100000)
        return int(value)
    except Exception:
        return None

def extract_filters(prompt: str) -> Dict[str, Any]:
    """Extract structured filters from a natural-language prompt.
    Returns keys: sector, salary_min, salary_max, location, education_level
    """
    text = (prompt or "").strip()
    lower = text.lower()
    filters: Dict[str, Any] = {}

    # Sector: words before 'job'/'jobs' or first noun-like token
    sector_match = re.search(r"(^|\b)([a-zA-Z]+)\s+jobs?\b", lower)
    if sector_match:
        filters["sector"] = sector_match.group(2)

    # Location: in <City/State>
    loc_match = re.search(r"\bin\s+([a-zA-Z\- ]{2,})\b", text)
    if loc_match:
        filters["location"] = loc_match.group(1).strip()

    # Education level
    if re.search(r"\b10th\b|matric", lower):
        filters["education_level"] = "10th"
    elif re.search(r"\b12th\b|higher\s*secondary|intermediate", lower):
        filters["education_level"] = "12th"
    elif re.search(r"\bgraduate|bachelor|degree\b", lower):
        filters["education_level"] = "graduate"
    elif re.search(r"\bpost\s*graduate|master|mba|mca|mtech\b", lower):
        filters["education_level"] = "postgraduate"

    # Salary: above/under/between patterns
    # between X and Y
    m_between = re.search(r"between\s+([\d.,]+\s*(?:k|thousand|l|lakh|lac|lakhs|lacs)?)\s+and\s+([\d.,]+\s*(?:k|thousand|l|lakh|lac|lakhs|lacs)?)", lower)
    if m_between:
        n1 = _parse_number_with_suffix(m_between.group(1))
        n2 = _parse_number_with_suffix(m_between.group(2))
        if n1 is not None and n2 is not None:
            filters["salary_min"], filters["salary_max"] = min(n1, n2), max(n1, n2)
            return filters

    # above / over / greater than
    m_above = re.search(r"(above|over|greater\s*than|>=)\s+([\d.,]+\s*(?:k|thousand|l|lakh|lac|lakhs|lacs)?)", lower)
    if m_above:
        n = _parse_number_with_suffix(m_above.group(2))
        if n is not None:
            filters["salary_min"] = n

    # under / below / less than
    m_under = re.search(r"(under|below|less\s*than|<=)\s+([\d.,]+\s*(?:k|thousand|l|lakh|lac|lakhs|lacs)?)", lower)
    if m_under:
        n = _parse_number_with_suffix(m_under.group(2))
        if n is not None:
            filters["salary_max"] = n

    # trailing '10k', '25,000' without qualifier; infer context if words like 'salary'
    if ("salary" in lower or "pay" in lower) and ("salary_min" not in filters and "salary_max" not in filters):
        lone = re.search(r"([\d.,]+\s*(?:k|thousand|l|lakh|lac|lakhs|lacs)?)", lower)
        if lone:
            n = _parse_number_with_suffix(lone.group(1))
            if n is not None:
                filters["salary_min"] = n

    return filters

class RAGSystemManager:
    """Manager class for RAG system instances"""
    
    def __init__(self):
        self.rag_systems: Dict[str, GovernmentJobRAG] = {}
    
    def create_rag_system(self, 
                         name: str, 
                         config: RAGConfig,
                         data_dir: str = ".") -> GovernmentJobRAG:
        """Create a new RAG system instance"""
        
        if name in self.rag_systems:
            logger.warning(f"RAG system '{name}' already exists. Returning existing instance.")
            return self.rag_systems[name]
        
        rag_system = GovernmentJobRAG(config)
        rag_system.initialize(data_dir)
        
        self.rag_systems[name] = rag_system
        logger.info(f"Created RAG system: {name}")
        
        return rag_system
    
    def get_rag_system(self, name: str) -> Optional[GovernmentJobRAG]:
        """Get an existing RAG system"""
        return self.rag_systems.get(name)
    
    def list_rag_systems(self) -> List[str]:
        """List all available RAG systems"""
        return list(self.rag_systems.keys())
    
    def remove_rag_system(self, name: str):
        """Remove a RAG system"""
        if name in self.rag_systems:
            del self.rag_systems[name]
            logger.info(f"Removed RAG system: {name}")

def create_default_config() -> RAGConfig:
    """Create default configuration"""
    vector_db_path = os.getenv("VECTOR_DB_PATH", "./chroma_db")
    return RAGConfig(
        embedding_model="all-MiniLM-L6-v2",
        vector_db_path=vector_db_path,
        collection_name="government_jobs",
        llm_model="gpt-3.5-turbo",
        llm_temperature=0.7,
        max_tokens=1000,
        use_openai=False,  # Set to True if you have OpenAI API key
        openai_api_key=None,
        n_results=5
    )

if __name__ == "__main__":
    # Test the complete RAG system
    logger.info("Testing Complete RAG System...")
    
    # Create configuration
    config = create_default_config()
    
    # Initialize RAG system
    rag_system = GovernmentJobRAG(config)
    rag_system.initialize()
    
    # Test queries
    test_queries = [
        "What are the eligibility requirements for banking jobs?",
        "Tell me about UPSC civil services exam",
        "What government jobs are available for graduates?",
        "How to apply for SSC exams?",
        "What is the salary range for defense jobs?"
    ]
    
    print("Testing RAG System Queries:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 30)
        
        response = rag_system.query(query)
        print(f"Answer: {response.answer[:200]}...")
        print(f"Confidence: {response.confidence_score:.2f}")
        print(f"Sources: {len(response.sources)}")
        
        if response.relevant_jobs:
            print(f"Relevant Jobs: {len(response.relevant_jobs)}")
            for job in response.relevant_jobs[:2]:  # Show first 2
                metadata = job.get('metadata', {})
                print(f"  - {metadata.get('exam_name', 'N/A')}")
    
    # Test database stats
    stats = rag_system.get_database_stats()
    print(f"\nDatabase Stats: {stats}")
    
    # Test categories
    categories = rag_system.get_all_categories()
    print(f"\nAvailable Categories: {categories}")

    # Demonstration: natural-language filters integration
    demo_query = "medical jobs above 20k in Delhi"
    print("\nDemo Query with Filters:")
    print(demo_query)
    extracted = extract_filters(demo_query)
    print(f"Extracted Filters: {extracted}")
    demo_response = rag_system.query(demo_query)
    print(f"Answer (first 200 chars): {demo_response.answer[:200]}...")



