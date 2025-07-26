import requests
import json
import time
from datetime import datetime

def test_api():
    """Test the Profile Authenticity API"""
    base_url = "http://localhost:8000"
    
    print("🚀 Testing Profile Authenticity API")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1️⃣ Testing Health Endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"   Status: {health_data.get('status')}")
            print(f"   Models Loaded: {health_data.get('models_loaded')}")
            print(f"   Version: {health_data.get('version')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return
    
    # Test 2: Model Info
    print("\n2️⃣ Testing Model Info Endpoint...")
    try:
        response = requests.get(f"{base_url}/model-info")
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            model_info = response.json()
            print(f"   Model Type: {model_info.get('model_type')}")
            print(f"   Feature Count: {model_info.get('feature_count')}")
            print(f"   Cross Validation: {model_info.get('cross_validation')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Single Profile Validation - Real Profile
    print("\n3️⃣ Testing Single Profile Validation - Real Profile...")
    real_profile = {
        "name": "Emily Chen",
        "email": "emily.chen@gmail.com",
        "address": "789 Pine Rd, Chicago, IL",
        "job_history": [
            {
                "title": "Software Engineer",
                "company": "Google",
                "start": 2018,
                "end": 2021
            },
            {
                "title": "Senior Software Engineer",
                "company": "Microsoft",
                "start": 2021,
                "end": 2024
            }
        ],
        "education": [
            {
                "school": "UC Berkeley",
                "degree": "BSc Computer Science"
            }
        ],
        "photo_flag": 0
    }
    
    try:
        response = requests.post(
            f"{base_url}/validate-profile",
            json=real_profile,
            headers={"Content-Type": "application/json"}
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Profile ID: {result.get('profile_id')}")
            print(f"   Authenticity Score: {result.get('authenticity_score'):.3f}")
            print(f"   Fake Probability: {result.get('fake_probability'):.3f}")
            print(f"   Risk Level: {result.get('risk_level')}")
            print(f"   Confidence: {result.get('confidence'):.3f}")
            print(f"   Processing Time: {result.get('processing_time_ms'):.1f}ms")
            print(f"   Flags: {result.get('flags')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Single Profile Validation - Fake Profile
    print("\n4️⃣ Testing Single Profile Validation - Fake Profile...")
    fake_profile = {
        "name": "James Bond",
        "email": "james.bond@mi6.gov.uk",
        "address": "007 Secret Street, London, UK",
        "job_history": [
            {
                "title": "Secret Agent",
                "company": "MI6",
                "start": 2000,
                "end": 2024
            }
        ],
        "education": [
            {
                "school": "Spy Academy",
                "degree": "Secret Agent Diploma"
            }
        ],
        "photo_flag": 1
    }
    
    try:
        response = requests.post(
            f"{base_url}/validate-profile",
            json=fake_profile,
            headers={"Content-Type": "application/json"}
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Profile ID: {result.get('profile_id')}")
            print(f"   Authenticity Score: {result.get('authenticity_score'):.3f}")
            print(f"   Fake Probability: {result.get('fake_probability'):.3f}")
            print(f"   Risk Level: {result.get('risk_level')}")
            print(f"   Confidence: {result.get('confidence'):.3f}")
            print(f"   Processing Time: {result.get('processing_time_ms'):.1f}ms")
            print(f"   Flags: {result.get('flags')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: Batch Validation
    print("\n5️⃣ Testing Batch Validation...")
    batch_profiles = [
        {
            "name": "John Smith",
            "email": "john.smith@gmail.com",
            "address": "123 Main St, New York, NY",
            "job_history": [
                {
                    "title": "Software Engineer",
                    "company": "Google",
                    "start": 2015,
                    "end": 2019
                }
            ],
            "education": [
                {
                    "school": "Harvard University",
                    "degree": "BSc Computer Science"
                }
            ],
            "photo_flag": 0
        },
        {
            "name": "Jane Doe",
            "email": "jane.doe@yahoo.com",
            "address": "456 Oak Ave, San Francisco, CA",
            "job_history": [
                {
                    "title": "Software Engineer",
                    "company": "Google",
                    "start": 2015,
                    "end": 2020
                },
                {
                    "title": "Senior Engineer",
                    "company": "Microsoft",
                    "start": 2018,
                    "end": 2023
                }
            ],
            "education": [
                {
                    "school": "Stanford University",
                    "degree": "BSc Computer Science"
                }
            ],
            "photo_flag": 0
        }
    ]
    
    try:
        response = requests.post(
            f"{base_url}/batch-validate",
            json={"profiles": batch_profiles},
            headers={"Content-Type": "application/json"}
        )
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Total Profiles: {result.get('summary', {}).get('total_profiles')}")
            print(f"   Avg Authenticity Score: {result.get('summary', {}).get('avg_authenticity_score'):.3f}")
            print(f"   High Risk Count: {result.get('summary', {}).get('high_risk_count')}")
            print(f"   Processing Time: {result.get('processing_time_ms'):.1f}ms")
            
            for i, profile_result in enumerate(result.get('results', [])):
                print(f"   Profile {i+1}: {profile_result.get('profile_id')} - Risk: {profile_result.get('risk_level')}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 6: Performance Test
    print("\n6️⃣ Testing Performance (10 profiles)...")
    test_profiles = []
    for i in range(10):
        profile = {
            "name": f"Test User {i}",
            "email": f"test.user{i}@gmail.com",
            "address": f"{i} Test St, Test City, TC",
            "job_history": [
                {
                    "title": "Software Engineer",
                    "company": "Test Company",
                    "start": 2020,
                    "end": 2024
                }
            ],
            "education": [
                {
                    "school": "Test University",
                    "degree": "BSc Computer Science"
                }
            ],
            "photo_flag": 0
        }
        test_profiles.append(profile)
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/batch-validate",
            json={"profiles": test_profiles},
            headers={"Content-Type": "application/json"}
        )
        end_time = time.time()
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            total_time = (end_time - start_time) * 1000
            api_time = result.get('processing_time_ms', 0)
            print(f"   Total Time: {total_time:.1f}ms")
            print(f"   API Processing Time: {api_time:.1f}ms")
            print(f"   Network Overhead: {total_time - api_time:.1f}ms")
            print(f"   Profiles per Second: {len(test_profiles) / (total_time / 1000):.1f}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ API Testing Completed!")

if __name__ == "__main__":
    test_api() 