#!/usr/bin/env python3
"""
Test script for the Government Job RAG system
"""

import os
import sys
import logging
from typing import List, Dict, Any
import json
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system import GovernmentJobRAG, RAGConfig, create_default_config
from data_processor import JobDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGTester:
    """Test class for the RAG system"""
    
    def __init__(self):
        self.rag_system = None
        self.test_results = []
    
    def initialize_system(self):
        """Initialize the RAG system"""
        logger.info("Initializing RAG system for testing...")
        
        try:
            config = create_default_config()
            self.rag_system = GovernmentJobRAG(config)
            self.rag_system.initialize()
            logger.info("RAG system initialized successfully!")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return False
    
    def test_basic_queries(self):
        """Test basic query functionality"""
        logger.info("Testing basic queries...")
        
        test_queries = [
            {
                "query": "What are the eligibility requirements for banking jobs?",
                "expected_keywords": ["graduate", "banking", "eligibility"]
            },
            {
                "query": "Tell me about UPSC civil services exam",
                "expected_keywords": ["upsc", "civil services", "ias", "ips"]
            },
            {
                "query": "What government jobs are available for graduates?",
                "expected_keywords": ["graduate", "government", "jobs"]
            },
            {
                "query": "How to apply for SSC exams?",
                "expected_keywords": ["ssc", "apply", "application"]
            },
            {
                "query": "What is the salary range for defense jobs?",
                "expected_keywords": ["defense", "salary", "₹"]
            }
        ]
        
        results = []
        for test_case in test_queries:
            logger.info(f"Testing query: {test_case['query']}")
            
            start_time = time.time()
            response = self.rag_system.query(test_case['query'])
            end_time = time.time()
            
            # Check if response contains expected keywords
            answer_lower = response.answer.lower()
            keyword_matches = sum(1 for keyword in test_case['expected_keywords'] 
                                if keyword.lower() in answer_lower)
            
            result = {
                "query": test_case['query'],
                "response_time": end_time - start_time,
                "confidence_score": response.confidence_score,
                "num_relevant_jobs": len(response.relevant_jobs),
                "num_sources": len(response.sources),
                "keyword_matches": keyword_matches,
                "total_expected_keywords": len(test_case['expected_keywords']),
                "success": keyword_matches > 0,
                "answer_preview": response.answer[:200] + "..." if len(response.answer) > 200 else response.answer
            }
            
            results.append(result)
            logger.info(f"Query completed in {result['response_time']:.2f}s, confidence: {result['confidence_score']:.2f}")
        
        self.test_results.extend(results)
        return results
    
    def test_filtered_queries(self):
        """Test queries with filters"""
        logger.info("Testing filtered queries...")
        
        filtered_tests = [
            {
                "query": "What banking jobs are available?",
                "category": "Banking",
                "expected_organizations": ["IBPS", "SBI", "RBI"]
            },
            {
                "query": "Jobs for graduates",
                "education_level": "graduate",
                "expected_keywords": ["graduate", "degree"]
            },
            {
                "query": "High salary government jobs",
                "salary_range": (50000, 200000),
                "expected_keywords": ["₹", "salary"]
            }
        ]
        
        results = []
        for test_case in filtered_tests:
            logger.info(f"Testing filtered query: {test_case['query']}")
            
            start_time = time.time()
            response = self.rag_system.query(
                question=test_case['query'],
                category=test_case.get('category'),
                education_level=test_case.get('education_level'),
                salary_range=test_case.get('salary_range')
            )
            end_time = time.time()
            
            # Check if results match filters
            filter_success = True
            if 'expected_organizations' in test_case:
                orgs_found = any(org in response.answer for org in test_case['expected_organizations'])
                filter_success = filter_success and orgs_found
            
            if 'expected_keywords' in test_case:
                keywords_found = any(keyword.lower() in response.answer.lower() 
                                  for keyword in test_case['expected_keywords'])
                filter_success = filter_success and keywords_found
            
            result = {
                "query": test_case['query'],
                "filters": {k: v for k, v in test_case.items() if k not in ['query', 'expected_organizations', 'expected_keywords']},
                "response_time": end_time - start_time,
                "confidence_score": response.confidence_score,
                "num_relevant_jobs": len(response.relevant_jobs),
                "filter_success": filter_success,
                "success": filter_success and response.confidence_score > 0.3
            }
            
            results.append(result)
            logger.info(f"Filtered query completed in {result['response_time']:.2f}s")
        
        self.test_results.extend(results)
        return results
    
    def test_recommendations(self):
        """Test job recommendation functionality"""
        logger.info("Testing job recommendations...")
        
        test_profiles = [
            {
                "education_level": "graduate",
                "interests": ["banking", "finance"],
                "experience_level": "entry"
            },
            {
                "education_level": "12th",
                "interests": ["defense", "security"],
                "experience_level": "entry"
            },
            {
                "education_level": "postgraduate",
                "interests": ["administration", "policy"],
                "experience_level": "experienced"
            }
        ]
        
        results = []
        for profile in test_profiles:
            logger.info(f"Testing recommendations for profile: {profile}")
            
            start_time = time.time()
            response = self.rag_system.get_job_recommendations(profile)
            end_time = time.time()
            
            result = {
                "profile": profile,
                "response_time": end_time - start_time,
                "confidence_score": response.confidence_score,
                "num_recommendations": len(response.relevant_jobs),
                "success": response.confidence_score > 0.3 and len(response.relevant_jobs) > 0
            }
            
            results.append(result)
            logger.info(f"Recommendations completed in {result['response_time']:.2f}s")
        
        self.test_results.extend(results)
        return results
    
    def test_search_functionality(self):
        """Test search functionality"""
        logger.info("Testing search functionality...")
        
        search_tests = [
            {
                "method": "search_by_category",
                "params": {"category": "Banking"},
                "expected_min_results": 5
            },
            {
                "method": "search_by_education",
                "params": {"education": "graduate"},
                "expected_min_results": 10
            },
            {
                "method": "get_jobs_by_organization",
                "params": {"organization": "SBI"},
                "expected_min_results": 1
            }
        ]
        
        results = []
        for test_case in search_tests:
            logger.info(f"Testing {test_case['method']} with params: {test_case['params']}")
            
            start_time = time.time()
            
            try:
                method = getattr(self.rag_system, test_case['method'])
                search_results = method(**test_case['params'])
                end_time = time.time()
                
                result = {
                    "method": test_case['method'],
                    "params": test_case['params'],
                    "response_time": end_time - start_time,
                    "num_results": len(search_results),
                    "expected_min_results": test_case['expected_min_results'],
                    "success": len(search_results) >= test_case['expected_min_results']
                }
                
                results.append(result)
                logger.info(f"Search completed in {result['response_time']:.2f}s, found {result['num_results']} results")
                
            except Exception as e:
                logger.error(f"Error in search test: {e}")
                results.append({
                    "method": test_case['method'],
                    "params": test_case['params'],
                    "error": str(e),
                    "success": False
                })
        
        self.test_results.extend(results)
        return results
    
    def test_database_stats(self):
        """Test database statistics"""
        logger.info("Testing database statistics...")
        
        try:
            stats = self.rag_system.get_database_stats()
            categories = self.rag_system.get_all_categories()
            
            result = {
                "test": "database_stats",
                "total_jobs": stats.get('total_jobs', 0),
                "total_categories": len(categories),
                "categories": categories,
                "success": stats.get('total_jobs', 0) > 0
            }
            
            logger.info(f"Database contains {result['total_jobs']} jobs across {result['total_categories']} categories")
            return [result]
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return [{"test": "database_stats", "error": str(e), "success": False}]
    
    def run_all_tests(self):
        """Run all tests"""
        logger.info("Starting comprehensive RAG system testing...")
        
        if not self.initialize_system():
            logger.error("Failed to initialize RAG system. Aborting tests.")
            return False
        
        # Run all test suites
        test_suites = [
            ("Basic Queries", self.test_basic_queries),
            ("Filtered Queries", self.test_filtered_queries),
            ("Recommendations", self.test_recommendations),
            ("Search Functionality", self.test_search_functionality),
            ("Database Stats", self.test_database_stats)
        ]
        
        all_results = []
        for suite_name, test_method in test_suites:
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {suite_name} Tests")
            logger.info(f"{'='*50}")
            
            try:
                results = test_method()
                all_results.extend(results)
                logger.info(f"{suite_name} tests completed successfully")
            except Exception as e:
                logger.error(f"Error in {suite_name} tests: {e}")
                all_results.append({
                    "test_suite": suite_name,
                    "error": str(e),
                    "success": False
                })
        
        # Generate summary
        self.generate_test_summary(all_results)
        
        # Save detailed results
        self.save_test_results(all_results)
        
        return True
    
    def generate_test_summary(self, results: List[Dict[str, Any]]):
        """Generate test summary"""
        logger.info(f"\n{'='*60}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*60}")
        
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.get('success', False))
        failed_tests = total_tests - successful_tests
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Successful: {successful_tests}")
        logger.info(f"Failed: {failed_tests}")
        logger.info(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        
        # Show failed tests
        if failed_tests > 0:
            logger.info(f"\nFailed Tests:")
            for result in results:
                if not result.get('success', False):
                    logger.info(f"  - {result.get('query', result.get('method', result.get('test', 'Unknown')))}")
        
        # Performance metrics
        response_times = [r.get('response_time', 0) for r in results if 'response_time' in r]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)
            max_response_time = max(response_times)
            logger.info(f"\nPerformance Metrics:")
            logger.info(f"  Average Response Time: {avg_response_time:.2f}s")
            logger.info(f"  Max Response Time: {max_response_time:.2f}s")
    
    def save_test_results(self, results: List[Dict[str, Any]]):
        """Save test results to file"""
        try:
            with open('test_results.json', 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info("Test results saved to test_results.json")
        except Exception as e:
            logger.error(f"Error saving test results: {e}")

def main():
    """Main test function"""
    tester = RAGTester()
    
    try:
        success = tester.run_all_tests()
        if success:
            logger.info("\nAll tests completed successfully!")
            sys.exit(0)
        else:
            logger.error("\nTests failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()



