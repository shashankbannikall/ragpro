from flask import Flask, render_template, request, jsonify, redirect, url_for
import json
import logging
from typing import Dict, Any, List
import os

from rag_system import GovernmentJobRAG, RAGConfig, create_default_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

# Global RAG system instance
rag_system = None

def initialize_rag_system():
    """Initialize the RAG system"""
    global rag_system
    try:
        config = create_default_config()
        rag_system = GovernmentJobRAG(config)
        rag_system.initialize()
        logger.info("RAG system initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        return False

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query_api():
    """API endpoint for querying the RAG system"""
    global rag_system
    
    if not rag_system:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        data = request.get_json()
        logger.info(f"Received query data: {data}")
        
        question = data.get('question', '')
        category = data.get('category', '')
        education_level = data.get('education_level', '')
        salary_range = data.get('salary_range', [])
        n_results = data.get('n_results', 5)
        
        logger.info(f"Processing query: {question}")
        
        # Convert salary range if provided
        salary_tuple = None
        if salary_range and len(salary_range) == 2:
            salary_tuple = (salary_range[0], salary_range[1])
        
        # Query the RAG system
        response = rag_system.query(
            question=question,
            category=category if category else None,
            education_level=education_level if education_level else None,
            salary_range=salary_tuple,
            n_results=n_results
        )
        
        logger.info(f"Generated response with confidence: {response.confidence_score}")
        
        result = {
            'answer': response.answer,
            'relevant_jobs': response.relevant_jobs,
            'sources': response.sources,
            'confidence_score': response.confidence_score,
            'query': question
        }
        
        logger.info(f"Returning result with {len(response.relevant_jobs)} jobs")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recommendations', methods=['POST'])
def recommendations_api():
    """API endpoint for getting job recommendations"""
    global rag_system
    
    if not rag_system:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        data = request.get_json()
        logger.info(f"Received recommendations data: {data}")
        
        profile = {
            'education_level': data.get('education_level', ''),
            'interests': data.get('interests', []),
            'experience_level': data.get('experience_level', ''),
            'location': data.get('location', ''),
            'salary_expectations': data.get('salary_expectations', '')
        }
        
        response = rag_system.get_job_recommendations(profile)
        
        result = {
            'answer': response.answer,
            'relevant_jobs': response.relevant_jobs,
            'sources': response.sources,
            'confidence_score': response.confidence_score
        }
        
        logger.info(f"Returning recommendations with {len(response.relevant_jobs)} jobs")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['POST'])
def search_api():
    """API endpoint for searching jobs"""
    global rag_system
    
    if not rag_system:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        data = request.get_json()
        logger.info(f"Received search data: {data}")
        
        query = data.get('query', '')
        n_results = data.get('n_results', 10)
        
        results = rag_system.vector_db.search_similar_jobs(query, n_results)
        
        result = {
            'jobs': results,
            'total_found': len(results),
            'query': query
        }
        
        logger.info(f"Returning search results with {len(results)} jobs")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error searching jobs: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/categories')
def categories_api():
    """API endpoint for getting job categories"""
    global rag_system
    
    if not rag_system:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        categories = rag_system.get_all_categories()
        return jsonify({
            'categories': categories,
            'total_categories': len(categories)
        })
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stats')
def stats_api():
    """API endpoint for getting database statistics"""
    global rag_system
    
    if not rag_system:
        return jsonify({'error': 'RAG system not initialized'}), 500
    
    try:
        stats = rag_system.get_database_stats()
        categories = rag_system.get_all_categories()
        
        stats.update({
            'total_categories': len(categories),
            'categories': categories
        })
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_api():
    """Health check endpoint"""
    global rag_system
    
    if not rag_system:
        return jsonify({'status': 'unhealthy', 'message': 'RAG system not initialized'}), 503
    
    try:
        stats = rag_system.get_database_stats()
        return jsonify({
            'status': 'healthy',
            'rag_system': 'initialized',
            'total_jobs': stats.get('total_jobs', 0)
        })
    except Exception as e:
        return jsonify({'status': 'unhealthy', 'message': str(e)}), 503

if __name__ == '__main__':
    # Initialize RAG system
    if initialize_rag_system():
        print("Starting Flask web UI...")
        print("Visit http://localhost:5000 to access the RAG system")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to initialize RAG system. Exiting...")



