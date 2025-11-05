#!/usr/bin/env python3
"""
Simple test script for the RAG system
"""

from vector_database import VectorDatabase
from data_processor import JobDataProcessor
from rag_generator import SimpleRAGGenerator

def test_basic_functionality():
    print("Testing Government Job RAG System...")
    print("=" * 50)
    
    # Test data loading
    print("1. Testing data loading...")
    processor = JobDataProcessor()
    jobs = processor.load_all_jobs()
    print(f"   ✓ Loaded {len(jobs)} jobs")
    
    # Test vector database
    print("2. Testing vector database...")
    vector_db = VectorDatabase()
    stats = vector_db.get_database_stats()
    print(f"   ✓ Database contains {stats.get('total_jobs', 0)} jobs")
    
    # Test search
    print("3. Testing search functionality...")
    results = vector_db.search_similar_jobs('banking jobs', n_results=3)
    print(f"   ✓ Found {len(results)} banking jobs:")
    for i, result in enumerate(results, 1):
        metadata = result['metadata']
        print(f"     {i}. {metadata['exam_name']} - {metadata['organization']}")
    
    # Test RAG generation
    print("4. Testing RAG generation...")
    generator = SimpleRAGGenerator()
    response = generator.generate_response("What are the eligibility requirements for banking jobs?", results)
    print(f"   ✓ Generated response (confidence: {response.confidence_score:.2f})")
    print(f"   Response preview: {response.answer[:150]}...")
    
    print("\n" + "=" * 50)
    print("All basic tests passed! ✓")
    print("The RAG system is working correctly.")

if __name__ == "__main__":
    test_basic_functionality()



