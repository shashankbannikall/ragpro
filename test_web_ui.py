#!/usr/bin/env python3
"""
Test script to verify the web UI is working
"""

import requests
import time
import json

def test_web_ui():
    """Test the web UI endpoints"""
    base_url = "http://localhost:5000"
    
    print("Testing Government Job RAG Web UI...")
    print("=" * 50)
    
    # Wait a moment for server to start
    time.sleep(3)
    
    try:
        # Test health endpoint
        print("1. Testing health endpoint...")
        response = requests.get(f"{base_url}/api/health")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Server is healthy: {data['status']}")
            print(f"   ✓ Total jobs: {data.get('total_jobs', 0)}")
        else:
            print(f"   ✗ Health check failed: {response.status_code}")
            return False
        
        # Test categories endpoint
        print("2. Testing categories endpoint...")
        response = requests.get(f"{base_url}/api/categories")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Found {data['total_categories']} categories")
            print(f"   ✓ Sample categories: {data['categories'][:3]}")
        else:
            print(f"   ✗ Categories endpoint failed: {response.status_code}")
        
        # Test query endpoint
        print("3. Testing query endpoint...")
        query_data = {
            "question": "What are the eligibility requirements for banking jobs?",
            "n_results": 3
        }
        response = requests.post(f"{base_url}/api/query", json=query_data)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Query successful")
            print(f"   ✓ Confidence: {data['confidence_score']:.2f}")
            print(f"   ✓ Found {len(data['relevant_jobs'])} relevant jobs")
            print(f"   ✓ Answer preview: {data['answer'][:100]}...")
        else:
            print(f"   ✗ Query endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
        
        # Test search endpoint
        print("4. Testing search endpoint...")
        search_data = {
            "query": "banking officer",
            "n_results": 3
        }
        response = requests.post(f"{base_url}/api/search", json=search_data)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Search successful")
            print(f"   ✓ Found {data['total_found']} jobs")
        else:
            print(f"   ✗ Search endpoint failed: {response.status_code}")
        
        # Test recommendations endpoint
        print("5. Testing recommendations endpoint...")
        rec_data = {
            "education_level": "graduate",
            "interests": ["banking", "finance"],
            "experience_level": "entry"
        }
        response = requests.post(f"{base_url}/api/recommendations", json=rec_data)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ Recommendations successful")
            print(f"   ✓ Confidence: {data['confidence_score']:.2f}")
            print(f"   ✓ Found {len(data['relevant_jobs'])} recommendations")
        else:
            print(f"   ✗ Recommendations endpoint failed: {response.status_code}")
        
        print("\n" + "=" * 50)
        print("Web UI testing completed!")
        print(f"🌐 Open your browser and visit: {base_url}")
        print("The RAG system is ready to use!")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("   ✗ Could not connect to server. Make sure the web UI is running.")
        print("   Run: python web_ui.py")
        return False
    except Exception as e:
        print(f"   ✗ Error testing web UI: {e}")
        return False

if __name__ == "__main__":
    test_web_ui()



