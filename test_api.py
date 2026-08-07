#!/usr/bin/env python3
"""
Test script for RoViT-KAN FastAPI
Verifies the API is working correctly
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import requests
import json
from datetime import datetime


def test_health():
    """Test health endpoint"""
    print("\n" + "=" * 60)
    print("Testing Health Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get('http://localhost:8000/health')
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_model_info():
    """Test model info endpoint"""
    print("\n" + "=" * 60)
    print("Testing Model Info Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get('http://localhost:8000/model-info')
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Model: {data.get('model_name', 'N/A')}")
        print(f"Classes: {data.get('num_classes', 'N/A')}")
        print(f"Device: {data.get('device', 'N/A')}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_classes():
    """Test classes endpoint"""
    print("\n" + "=" * 60)
    print("Testing Classes Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get('http://localhost:8000/classes')
        print(f"Status: {response.status_code}")
        data = response.json()
        print(f"Classes: {data.get('classes', [])}")
        print(f"Severity Map: {data.get('severity_map', {})}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_web_ui():
    """Test web UI endpoint"""
    print("\n" + "=" * 60)
    print("Testing Web UI Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get('http://localhost:8000/')
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('content-type', 'N/A')}")
        
        if 'text/html' in response.headers.get('content-type', ''):
            # Check if it contains expected elements
            if 'RoViT-KAN' in response.text:
                print("✅ HTML content verified")
                return True
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("🌹 RoViT-KAN API Test Suite")
    print("=" * 60)
    print(f"Time: {datetime.now().isoformat()}")
    print("Make sure the server is running: python run_api.py")
    
    results = []
    
    # Run tests
    results.append(("Health", test_health()))
    results.append(("Model Info", test_model_info()))
    results.append(("Classes", test_classes()))
    results.append(("Web UI", test_web_ui()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20s} {status}")
    
    print("=" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
