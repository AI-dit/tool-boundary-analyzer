#!/usr/bin/env python3
"""
Test script for the tool-boundary-analyzer package
"""

def test_imports():
    """Test that all imports work correctly"""
    try:
        from tool_visualizer import create_app, run_server, cli_main, __version__
        print("✅ Import test passed")
        print(f"📦 Package version: {__version__}")
        return True
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_app_creation():
    """Test that we can create a Flask app"""
    try:
        from tool_visualizer import create_app
        app = create_app()
        print("✅ App creation test passed")
        print(f"🌐 Flask app: {app}")
        return True
    except Exception as e:
        print(f"❌ App creation test failed: {e}")
        return False

def test_health_endpoint():
    """Test the health endpoint"""
    try:
        from tool_visualizer import create_app
        import json
        
        app = create_app()
        with app.test_client() as client:
            response = client.get('/health')
            data = json.loads(response.data)
            
            if response.status_code == 200 and data['status'] == 'healthy':
                print("✅ Health endpoint test passed")
                print(f"🏥 Health response: {data}")
                return True
            else:
                print(f"❌ Health endpoint test failed: {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Health endpoint test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing tool-boundary-analyzer package...")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_app_creation,
        test_health_endpoint,
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    passed = sum(results)
    total = len(results)
    
    print("=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Package is working correctly.")
        return 0
    else:
        print("⚠️ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    exit(main())
