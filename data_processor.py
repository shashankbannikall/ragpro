import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class JobData:
    """Data class for job information"""
    id: str
    category: str
    subcategory: str
    exam_name: str
    organization: str
    frequency: str
    eligibility: str
    age_limit: str
    selection_process: str
    job_roles: List[str]
    tags: List[str]
    official_website: str
    description: str
    salary_range: str
    application_cycle: str
    level: str
    status: str

class JobDataProcessor:
    """Processes and loads government job data from JSON files"""
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = Path(data_dir)
        self.jobs: List[JobData] = []
        
    def load_all_jobs(self) -> List[JobData]:
        """Load all job data from JSON files"""
        json_files = [
            "admin_jobs.json",
            "auto_jobs.json", 
            "banking_jobs.json",
            "defence_jobs.json",
            "health_jobs.json",
            "judiciary_jobs.json",
            "nta_jobs.json",
            "psu.json",
            "research_jobs.json",
            "rrb_exams.json",
            "ssc_exams.json",
            "upsc_jobs.json"
        ]
        
        all_jobs = []
        
        for json_file in json_files:
            file_path = self.data_dir / json_file
            if file_path.exists():
                logger.info(f"Loading jobs from {json_file}")
                jobs = self._load_jobs_from_file(file_path)
                all_jobs.extend(jobs)
            else:
                logger.warning(f"File {json_file} not found, skipping...")
                
        self.jobs = all_jobs
        logger.info(f"Total jobs loaded: {len(all_jobs)}")
        return all_jobs
    
    def _load_jobs_from_file(self, file_path: Path) -> List[JobData]:
        """Load jobs from a single JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            jobs = []
            for job_dict in data:
                try:
                    job = JobData(**job_dict)
                    jobs.append(job)
                except Exception as e:
                    logger.error(f"Error processing job {job_dict.get('id', 'unknown')}: {e}")
                    continue
                    
            return jobs
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return []
    
    def create_searchable_text(self, job: JobData) -> str:
        """Create searchable text from job data"""
        text_parts = [
            f"Job: {job.exam_name}",
            f"Organization: {job.organization}",
            f"Category: {job.category}",
            f"Subcategory: {job.subcategory}",
            f"Description: {job.description}",
            f"Eligibility: {job.eligibility}",
            f"Job Roles: {', '.join(job.job_roles)}",
            f"Tags: {', '.join(job.tags)}",
            f"Salary: {job.salary_range}",
            f"Age Limit: {job.age_limit}",
            f"Selection Process: {job.selection_process}",
            f"Application Cycle: {job.application_cycle}"
        ]
        
        return " | ".join(text_parts)
    
    def get_jobs_by_category(self, category: str) -> List[JobData]:
        """Filter jobs by category"""
        return [job for job in self.jobs if category.lower() in job.category.lower()]
    
    def get_jobs_by_tags(self, tags: List[str]) -> List[JobData]:
        """Filter jobs by tags"""
        matching_jobs = []
        for job in self.jobs:
            if any(tag.lower() in [t.lower() for t in job.tags] for tag in tags):
                matching_jobs.append(job)
        return matching_jobs
    
    def get_jobs_by_education_level(self, education: str) -> List[JobData]:
        """Filter jobs by education level"""
        education_keywords = {
            "10th": ["10th", "matriculation", "10th pass"],
            "12th": ["12th", "higher secondary", "12th pass", "intermediate"],
            "graduate": ["graduate", "bachelor", "degree", "graduation"],
            "postgraduate": ["postgraduate", "master", "mba", "mca", "mtech"]
        }
        
        matching_jobs = []
        for job in self.jobs:
            eligibility_lower = job.eligibility.lower()
            if any(keyword in eligibility_lower for keyword in education_keywords.get(education.lower(), [])):
                matching_jobs.append(job)
        
        return matching_jobs
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert jobs to pandas DataFrame"""
        data = []
        for job in self.jobs:
            data.append({
                'id': job.id,
                'category': job.category,
                'subcategory': job.subcategory,
                'exam_name': job.exam_name,
                'organization': job.organization,
                'frequency': job.frequency,
                'eligibility': job.eligibility,
                'age_limit': job.age_limit,
                'selection_process': job.selection_process,
                'job_roles': ', '.join(job.job_roles),
                'tags': ', '.join(job.tags),
                'official_website': job.official_website,
                'description': job.description,
                'salary_range': job.salary_range,
                'application_cycle': job.application_cycle,
                'level': job.level,
                'status': job.status
            })
        
        return pd.DataFrame(data)

if __name__ == "__main__":
    # Test the data processor
    processor = JobDataProcessor()
    jobs = processor.load_all_jobs()
    
    print(f"Loaded {len(jobs)} jobs")
    print(f"Categories: {set(job.category for job in jobs)}")
    
    # Test searchable text creation
    if jobs:
        sample_job = jobs[0]
        searchable_text = processor.create_searchable_text(sample_job)
        print(f"\nSample searchable text:\n{searchable_text[:200]}...")

