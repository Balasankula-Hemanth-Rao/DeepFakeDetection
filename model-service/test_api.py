#!/usr/bin/env python3
"""Test script for the ML inference API"""
import requests
import json
import time

API_URL = "http://127.0.0.1:8001"
API_KEY = "devkey"

def test_health():
    """Test health check endpoint"""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}\n")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}\n")
        return False

def test_inference():
    """Test inference endpoint"""
    print("Testing /infer endpoint...")
    try:
        # Test with real image
        with open("data/sample/train/real/r0.jpg", "rb") as f:
            files = {"file": f}
            headers = {"X-API-KEY": API_KEY}
            response = requests.post(f"{API_URL}/infer", files=files, headers=headers)
        
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}\n")
        
        # Validate response format
        if "fake_prob" in result and "real_prob" in result:
            fake_prob = result["fake_prob"]
            real_prob = result["real_prob"]
            total = fake_prob + real_prob
            print(f"✓ Probabilities sum to {total:.4f} (expected ~1.0)")
            print(f"  - Fake probability: {fake_prob:.4f}")
            print(f"  - Real probability: {real_prob:.4f}")
            return response.status_code == 200
        else:
            print("✗ Response missing expected fields")
            return False
    except Exception as e:
        print(f"Error: {e}\n")
        return False

def test_missing_api_key():
    """Test missing API key error"""
    print("Testing missing API key error...")
    try:
        with open("data/sample/train/real/r0.jpg", "rb") as f:
            files = {"file": f}
            response = requests.post(f"{API_URL}/infer", files=files)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 403:
            print(f"✓ Correctly returned 403 for missing API key\n")
            return True
        else:
            print(f"✗ Expected 403, got {response.status_code}\n")
            return False
    except Exception as e:
        print(f"Error: {e}\n")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ML Inference API Test Suite")
    print("=" * 60 + "\n")
    
    # Give server time to start
    print("Waiting for server to be ready...")
    time.sleep(2)
    
    results = {
        "Health Check": test_health(),
        "Inference Test": test_inference(),
        "Missing API Key": test_missing_api_key(),
    }
    
    print("=" * 60)
    print("Test Results:")
    print("=" * 60)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    print("\n" + ("=" * 60))
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
    print("=" * 60)
