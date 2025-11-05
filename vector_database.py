import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
import re
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging
from pathlib import Path

from data_processor import JobDataProcessor, JobData

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Manages vector embeddings and similarity search for job data"""
    
    def __init__(self, 
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "government_jobs"):
        
        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model_name}")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        # Initialize ChromaDB
        self._init_chroma_db()
        
    def _init_chroma_db(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create ChromaDB client (persistent or in-memory) based on env
            use_persist = os.getenv("RAG_PERSIST", "true").lower() == "true"
            persist_path = os.getenv("VECTOR_DB_PATH", self.persist_directory)
            if use_persist:
                self.chroma_client = chromadb.PersistentClient(
                    path=persist_path,
                    settings=Settings(anonymized_telemetry=False)
                )
            else:
                self.chroma_client = chromadb.Client(
                    Settings(anonymized_telemetry=False)
                )
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(
                    name=self.collection_name
                )
                logger.info(f"Loaded existing collection: {self.collection_name}")
            except:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Government job database"}
                )
                logger.info(f"Created new collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {e}")
            raise
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts"""
        try:
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            logger.error(f"Error creating embeddings: {e}")
            raise
    
    def add_jobs_to_database(self, jobs: List[JobData], processor: JobDataProcessor):
        """Add jobs to the vector database"""
        logger.info(f"Adding {len(jobs)} jobs to vector database...")
        
        # Prepare data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for job in jobs:
            # Create searchable text
            searchable_text = processor.create_searchable_text(job)
            
            # Prepare metadata
            metadata = {
                "id": job.id,
                "category": job.category,
                "subcategory": job.subcategory,
                "exam_name": job.exam_name,
                "organization": job.organization,
                "frequency": job.frequency,
                "eligibility": job.eligibility,
                "age_limit": job.age_limit,
                "selection_process": job.selection_process,
                "job_roles": ", ".join(job.job_roles),
                "tags": ", ".join(job.tags),
                "official_website": job.official_website,
                "description": job.description,
                "salary_range": job.salary_range,
                "application_cycle": job.application_cycle,
                "level": job.level,
                "status": job.status
            }
            
            ids.append(job.id)
            documents.append(searchable_text)
            metadatas.append(metadata)
        
        try:
            # Add to ChromaDB collection
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Successfully added {len(jobs)} jobs to database")
            
        except Exception as e:
            logger.error(f"Error adding jobs to database: {e}")
            raise
    
    def search_similar_jobs(self, 
                           query: str, 
                           n_results: int = 5,
                           filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar jobs based on query"""
        try:
            # Perform similarity search
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=filter_metadata
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching database: {e}")
            raise
    
    def search_by_category(self, category: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search jobs by category using semantic hint + client-side filtering."""
        # Fetch more candidates, then filter locally for robustness across DB versions
        candidates = self.search_similar_jobs(
            query=f"category: {category}",
            n_results=max(n_results * 4, 20),
            filter_metadata=None
        )
        filtered: List[Dict[str, Any]] = []
        key = (category or "").lower()
        for r in candidates:
            cat_val = (r.get('metadata', {}) or {}).get('category', '')
            if key and key not in cat_val.lower() and cat_val.lower() not in key:
                continue
            filtered.append(r)
        return filtered[:n_results]
    
    def search_by_education_level(self, education: str, n_results: int = 10) -> List[Dict[str, Any]]:
        """Search jobs by education level"""
        education_queries = {
            "10th": "10th pass matriculation",
            "12th": "12th pass higher secondary intermediate",
            "graduate": "graduate bachelor degree graduation",
            "postgraduate": "postgraduate master mba mca mtech"
        }
        
        query = education_queries.get(education.lower(), education)
        return self.search_similar_jobs(query, n_results)
    
    def search_by_salary_range(self, min_salary: Optional[int] = None, 
                             max_salary: Optional[int] = None,
                             n_results: int = 10) -> List[Dict[str, Any]]:
        """Search jobs by salary range (approximate)"""
        if min_salary and max_salary:
            query = f"salary range {min_salary} to {max_salary} rupees"
        elif min_salary:
            query = f"salary above {min_salary} rupees"
        elif max_salary:
            query = f"salary below {max_salary} rupees"
        else:
            query = "government job salary"
            
        return self.search_similar_jobs(query, n_results)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            count = self.collection.count()
            return {
                "total_jobs": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model_name,
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def clear_database(self):
        """Clear all data from the database"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self._init_chroma_db()
            logger.info("Database cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            raise

class JobRetriever:
    """High-level job retrieval interface"""
    
    def __init__(self, vector_db: VectorDatabase):
        self.vector_db = vector_db
    
    def find_jobs(self, 
                 query: str,
                 category: Optional[str] = None,
                 education_level: Optional[str] = None,
                 salary_range: Optional[Tuple[int, int]] = None,
                 n_results: int = 5) -> List[Dict[str, Any]]:
        """Find jobs based on multiple criteria"""
        
        # Build a semantic query that includes category hint if provided
        search_query = query
        if category:
            search_query = f"{query} category:{category}"
        
        # Perform search without strict metadata filter (to avoid operator/version issues),
        # then apply robust client-side filtering below.
        results = self.vector_db.search_similar_jobs(
            query=search_query,
            n_results=max(n_results * 4, 20),
            filter_metadata=None
        )
        
        # Additional filtering for education and salary
        filtered_results = []

        def parse_salary_text_to_range(s: str) -> Tuple[Optional[int], Optional[int]]:
            """Attempt to parse textual salary like '₹36,000 – ₹50,000' into numeric min/max in rupees."""
            if not s:
                return (None, None)
            t = s.lower().replace(",", " ")
            # capture two amounts optionally separated by dash or 'to'
            nums = [m.group(0) for m in re.finditer(r"\d+[\d\s]*", t)]
            vals: List[int] = []
            for n in nums:
                try:
                    vals.append(int(re.sub(r"\s+", "", n)))
                except Exception:
                    continue
            if len(vals) >= 2:
                a, b = sorted(vals[:2])
                return (a, b)
            if len(vals) == 1:
                return (vals[0], None)
            return (None, None)
        for result in results:
            metadata = result['metadata']
            
            # Check category match (case-insensitive, substring tolerant) across multiple fields
            if category:
                key = category.lower()
                cat_val = (metadata.get('category', '') or '').lower()
                sub_val = (metadata.get('subcategory', '') or '').lower()
                tags_val = (metadata.get('tags', '') or '').lower()
                name_val = (metadata.get('exam_name', '') or '').lower()
                if not (key in cat_val or key in sub_val or key in tags_val or key in name_val or cat_val in key):
                    continue

            # Check education level match
            if education_level:
                eligibility = metadata.get('eligibility', '').lower()
                education_keywords = {
                    "10th": ["10th", "matriculation"],
                    "12th": ["12th", "higher secondary", "intermediate"],
                    "graduate": ["graduate", "bachelor", "degree"],
                    "postgraduate": ["postgraduate", "master", "mba"]
                }
                
                if not any(keyword in eligibility for keyword in education_keywords.get(education_level.lower(), [])):
                    continue
            
            # Check salary range using numeric parse from text
            if salary_range:
                stated_min, stated_max = parse_salary_text_to_range(metadata.get('salary_range', ''))
                req_min, req_max = salary_range
                # If we can't parse, skip salary filtering
                if stated_min is not None or stated_max is not None:
                    if req_min is not None:
                        # require stated max >= req_min (overlap)
                        if (stated_max or stated_min or 0) < req_min:
                            continue
                    if req_max is not None:
                        # require stated min <= req_max
                        if (stated_min or stated_max or 10**12) > req_max:
                            continue
            
            filtered_results.append(result)
        
        return filtered_results[:n_results]
    
    def get_job_recommendations(self, 
                               user_profile: Dict[str, Any],
                               n_results: int = 5) -> List[Dict[str, Any]]:
        """Get job recommendations based on user profile"""
        
        # Extract user preferences
        education = user_profile.get('education_level', '')
        interests = user_profile.get('interests', [])
        experience = user_profile.get('experience_level', '')
        
        # Build query based on profile
        query_parts = []
        if education:
            query_parts.append(f"education: {education}")
        if interests:
            query_parts.append(f"interests: {', '.join(interests)}")
        if experience:
            query_parts.append(f"experience: {experience}")
        
        query = " ".join(query_parts) if query_parts else "government job"
        
        return self.vector_db.search_similar_jobs(query, n_results)

if __name__ == "__main__":
    # Test the vector database
    logger.info("Testing Vector Database...")
    
    # Initialize components
    processor = JobDataProcessor()
    vector_db = VectorDatabase()
    
    # Load jobs
    jobs = processor.load_all_jobs()
    
    # Add jobs to database
    vector_db.add_jobs_to_database(jobs, processor)
    
    # Test search
    results = vector_db.search_similar_jobs("banking officer job", n_results=3)
    print(f"Found {len(results)} similar jobs")
    
    for result in results:
        print(f"- {result['metadata']['exam_name']} ({result['metadata']['organization']})")
    
    # Get database stats
    stats = vector_db.get_database_stats()
    print(f"\nDatabase stats: {stats}")



