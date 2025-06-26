#!/usr/bin/env python3
"""
Test script for the FastAPI Toxic Classifier

This script demonstrates how to use the FastAPI endpoints
for toxic content classification.
"""

import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("🏥 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_root():
    """Test the root endpoint."""
    print("\n🏠 Testing Root Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False

def test_simple_classification(text: str):
    """Test the simple classification endpoint."""
    print(f"\n🔍 Testing Simple Classification: '{text[:50]}...'")
    try:
        response = requests.post(f"{BASE_URL}/classify/simple", json=text)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ Classification Results:")
            print(f"   User Input: {result['user_input']}")
            print(f"   Detected Labels: {result['detected_labels']}")
            print(f"   Search Results Found: {len(result['search_results'])}")
            print(f"   LLM Classification: {result['llm_response']['classification']}")
            print(f"   LLM Analysis Type: {result['llm_response']['analysis_type']}")
        else:
            print(f"❌ Classification failed: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Simple classification failed: {e}")
        return False

def test_full_classification(text: str, **kwargs):
    """Test the full classification endpoint with custom parameters."""
    print(f"\n🔬 Testing Full Classification: '{text[:50]}...'")
    try:
        payload = {
            "text": text,
            **kwargs
        }
        response = requests.post(f"{BASE_URL}/classify", json=payload)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print("✅ Full Classification Results:")
            print(f"   User Input: {result['user_input']}")
            print(f"   Detected Labels: {result['detected_labels']}")
            print(f"   Search Results Found: {len(result['search_results'])}")
            print(f"   LLM Classification: {result['llm_response']['classification']}")
            print(f"   LLM Analysis Type: {result['llm_response']['analysis_type']}")
            print(f"   Processing Info: {result['processing_info']}")
            
            # Show search results details
            if result['search_results']:
                print("\n   📊 Search Results Details:")
                for i, search_result in enumerate(result['search_results'][:2]):  # Show first 2
                    print(f"     {i+1}. Similarity: {search_result['similarity_score']:.3f}")
                    print(f"        Text: {search_result['text'][:100]}...")
                    print(f"        Labels: {search_result['labels']}")
        else:
            print(f"❌ Full classification failed: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Full classification failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 FastAPI Toxic Classifier Test Suite")
    print("=" * 50)
    
    # Test endpoints
    health_ok = test_health()
    root_ok = test_root()
    
    if not health_ok:
        print("\n❌ Health check failed. Make sure the API is running.")
        print("Start the API with: python main.py")
        return
    
    # Test cases
    test_cases = [
        "You are a stupid idiot and I hope you die",
        "This is a great product, buy now for 50% off!",
        "Women should stay in the kitchen and cook",
        "Hello, how are you today?",
        "I'm going to hurt you if you don't do what I say"
    ]
    
    print("\n" + "=" * 50)
    print("TESTING CLASSIFICATION ENDPOINTS")
    print("=" * 50)
    
    # Test simple classification
    print("\n📝 Testing Simple Classification Endpoint:")
    for i, text in enumerate(test_cases[:2], 1):
        print(f"\nTest Case {i}:")
        test_simple_classification(text)
        time.sleep(1)  # Small delay between requests
    
    # Test full classification with different parameters
    print("\n📝 Testing Full Classification Endpoint:")
    for i, text in enumerate(test_cases[2:4], 1):
        print(f"\nTest Case {i}:")
        test_full_classification(
            text,
            confidence_threshold=0.2,
            search_limit=5,
            use_local_llm=True
        )
        time.sleep(1)
    
    # Test with rule-based analysis (no LLM)
    print("\n📝 Testing with Rule-based Analysis:")
    test_full_classification(
        test_cases[4],
        confidence_threshold=0.3,
        search_limit=3,
        use_local_llm=False
    )
    
    print("\n✅ Test suite completed!")

if __name__ == "__main__":
    main() 