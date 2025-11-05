# Government Job RAG System - Setup and Usage Guide

## Overview
This is a comprehensive RAG (Retrieval-Augmented Generation) system for government job queries. It combines vector similarity search with natural language generation to provide intelligent responses about government job opportunities.

## Features
- **Vector-based Job Search**: Uses ChromaDB and sentence transformers for semantic search
- **Natural Language Queries**: Ask questions in plain English about government jobs
- **Filtered Search**: Search by category, education level, salary range, etc.
- **Personalized Recommendations**: Get job recommendations based on your profile
- **REST API**: Complete API interface for integration
- **Multiple LLM Support**: Works with OpenAI GPT models or simple template-based responses

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Optional: Set up OpenAI API Key (for advanced LLM features)
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Quick Start

### 1. Basic Usage
```python
from rag_system import GovernmentJobRAG, create_default_config

# Create configuration
config = create_default_config()

# Initialize RAG system
rag_system = GovernmentJobRAG(config)
rag_system.initialize()

# Query the system
response = rag_system.query("What are the eligibility requirements for banking jobs?")
print(response.answer)
```

### 2. Using the API Server
```bash
# Start the API server
python api_server.py

# The API will be available at http://localhost:8000
# Visit http://localhost:8000/docs for interactive API documentation
```

### 3. Running Tests
```bash
# Run comprehensive tests
python test_rag_system.py
```

## API Endpoints

### Query Jobs
```bash
POST /query
{
    "question": "What are the eligibility requirements for banking jobs?",
    "category": "Banking",
    "education_level": "graduate",
    "n_results": 5
}
```

### Get Recommendations
```bash
POST /recommendations
{
    "education_level": "graduate",
    "interests": ["banking", "finance"],
    "experience_level": "entry"
}
```

### Search Jobs
```bash
POST /search
{
    "query": "banking officer",
    "n_results": 10
}
```

### Get Categories
```bash
GET /categories
```

### Get Statistics
```bash
GET /stats
```

## Configuration

### RAGConfig Options
```python
config = RAGConfig(
    embedding_model="all-MiniLM-L6-v2",  # Sentence transformer model
    vector_db_path="./chroma_db",        # ChromaDB storage path
    collection_name="government_jobs",   # Collection name
    llm_model="gpt-3.5-turbo",          # OpenAI model (if using OpenAI)
    llm_temperature=0.7,                # LLM temperature
    max_tokens=1000,                    # Max tokens for response
    use_openai=False,                    # Whether to use OpenAI
    openai_api_key=None,                 # OpenAI API key
    n_results=5                          # Default number of results
)
```

## Data Structure

The system expects JSON files with the following structure:
```json
[
    {
        "id": "unique_id",
        "category": "Job Category",
        "subcategory": "Subcategory",
        "exam_name": "Exam Name",
        "organization": "Organization Name",
        "frequency": "Annual/Biennial/etc",
        "eligibility": "Eligibility requirements",
        "age_limit": "Age limit",
        "selection_process": "Selection process details",
        "job_roles": ["Role1", "Role2"],
        "tags": ["tag1", "tag2"],
        "official_website": "https://example.com",
        "description": "Job description",
        "salary_range": "Salary range",
        "application_cycle": "Application cycle",
        "level": "Central/State",
        "status": "Active/Inactive"
    }
]
```

## Usage Examples

### 1. Basic Query
```python
response = rag_system.query("What government jobs are available for graduates?")
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence_score}")
print(f"Sources: {response.sources}")
```

### 2. Filtered Search
```python
response = rag_system.query(
    question="Banking jobs with good salary",
    category="Banking",
    education_level="graduate",
    salary_range=(30000, 100000)
)
```

### 3. Get Recommendations
```python
profile = {
    "education_level": "graduate",
    "interests": ["banking", "finance"],
    "experience_level": "entry"
}
response = rag_system.get_job_recommendations(profile)
```

### 4. Search by Category
```python
banking_jobs = rag_system.search_by_category("Banking", n_results=10)
```

### 5. Search by Education
```python
graduate_jobs = rag_system.search_by_education("graduate", n_results=15)
```

## File Structure
```
├── requirements.txt          # Python dependencies
├── data_processor.py         # Data loading and preprocessing
├── vector_database.py        # Vector database and retrieval
├── rag_generator.py          # Response generation
├── rag_system.py            # Main RAG system
├── api_server.py            # FastAPI server
├── test_rag_system.py       # Test suite
├── setup_guide.md           # This file
└── chroma_db/               # Vector database storage (created automatically)
```

## Advanced Usage

### 1. Custom Embedding Model
```python
config = RAGConfig(embedding_model="all-mpnet-base-v2")
```

### 2. Using OpenAI for Better Responses
```python
config = RAGConfig(
    use_openai=True,
    openai_api_key="your-key",
    llm_model="gpt-4"
)
```

### 3. Custom Vector Database Path
```python
config = RAGConfig(vector_db_path="/path/to/custom/db")
```

## Troubleshooting

### Common Issues

1. **ChromaDB Permission Error**
   - Ensure write permissions for the vector database directory
   - Try running with different `vector_db_path`

2. **OpenAI API Errors**
   - Check API key validity
   - Ensure sufficient API credits
   - Fall back to simple generator by setting `use_openai=False`

3. **Memory Issues**
   - Reduce `n_results` parameter
   - Use smaller embedding model
   - Process data in batches

4. **No Results Found**
   - Check if JSON files are properly formatted
   - Verify data loading with `rag_system.get_database_stats()`
   - Try broader search terms

### Performance Optimization

1. **Faster Embeddings**
   - Use GPU if available: `pip install sentence-transformers[gpu]`
   - Use smaller models for faster processing

2. **Reduced Memory Usage**
   - Process data in smaller batches
   - Use FAISS instead of ChromaDB for large datasets

3. **Better Search Results**
   - Fine-tune embedding model on job data
   - Use hybrid search (keyword + semantic)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is open source. Please check the license file for details.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test suite to identify problems
3. Create an issue with detailed error information



