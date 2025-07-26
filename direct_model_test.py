from profile_authenticity_classifier import EnhancedProfileAuthenticityClassifier
import time

def test_model_directly():
    """Test the model directly without API"""
    print("🚀 Direct Model Testing")
    print("=" * 50)
    
    # Load the model
    print("📦 Loading model...")
    classifier = EnhancedProfileAuthenticityClassifier()
    classifier.load_models()
    print("✅ Model loaded successfully!")
    
    # Test 1: Obviously Fake Profile
    print("\n1️⃣ Testing Obviously Fake Profile...")
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
    
    start_time = time.time()
    result = classifier.predict_profile(fake_profile)
    processing_time = (time.time() - start_time) * 1000
    
    print(f"   Profile: {fake_profile['name']}")
    print(f"   Authenticity Score: {result['authenticity_score']:.3f}")
    print(f"   Fake Probability: {result['probability']:.3f}")
    print(f"   Prediction: {'FAKE' if result['probability'] > 0.5 else 'REAL'}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Processing Time: {processing_time:.1f}ms")
    print(f"   Individual Scores:")
    for model, score in result['individual_scores'].items():
        print(f"     {model}: {score:.3f}")
    
    # Test 2: Realistic Profile
    print("\n2️⃣ Testing Realistic Profile...")
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
    
    start_time = time.time()
    result = classifier.predict_profile(real_profile)
    processing_time = (time.time() - start_time) * 1000
    
    print(f"   Profile: {real_profile['name']}")
    print(f"   Authenticity Score: {result['authenticity_score']:.3f}")
    print(f"   Fake Probability: {result['probability']:.3f}")
    print(f"   Prediction: {'FAKE' if result['probability'] > 0.5 else 'REAL'}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Processing Time: {processing_time:.1f}ms")
    print(f"   Individual Scores:")
    for model, score in result['individual_scores'].items():
        print(f"     {model}: {score:.3f}")
    
    # Test 3: AI-Generated Content
    print("\n3️⃣ Testing AI-Generated Content...")
    ai_profile = {
        "name": "John Smith",
        "email": "john.smith@gmail.com",
        "address": "123 Main St, New York, NY",
        "job_history": [
            {
                "title": "Exceptional Senior Software Engineer",
                "company": "Outstanding Tech Solutions Inc",
                "start": 2018,
                "end": 2024
            }
        ],
        "education": [
            {
                "school": "Phenomenal University",
                "degree": "Brilliant Computer Science Degree"
            }
        ],
        "photo_flag": 0
    }
    
    start_time = time.time()
    result = classifier.predict_profile(ai_profile)
    processing_time = (time.time() - start_time) * 1000
    
    print(f"   Profile: {ai_profile['name']}")
    print(f"   Authenticity Score: {result['authenticity_score']:.3f}")
    print(f"   Fake Probability: {result['probability']:.3f}")
    print(f"   Prediction: {'FAKE' if result['probability'] > 0.5 else 'REAL'}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Processing Time: {processing_time:.1f}ms")
    print(f"   Individual Scores:")
    for model, score in result['individual_scores'].items():
        print(f"     {model}: {score:.3f}")
    
    # Test 4: Timeline Overlap
    print("\n4️⃣ Testing Timeline Overlap...")
    overlap_profile = {
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
    
    start_time = time.time()
    result = classifier.predict_profile(overlap_profile)
    processing_time = (time.time() - start_time) * 1000
    
    print(f"   Profile: {overlap_profile['name']}")
    print(f"   Authenticity Score: {result['authenticity_score']:.3f}")
    print(f"   Fake Probability: {result['probability']:.3f}")
    print(f"   Prediction: {'FAKE' if result['probability'] > 0.5 else 'REAL'}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Processing Time: {processing_time:.1f}ms")
    print(f"   Individual Scores:")
    for model, score in result['individual_scores'].items():
        print(f"     {model}: {score:.3f}")
    
    # Test 5: Performance Test
    print("\n5️⃣ Performance Test (10 profiles)...")
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
    
    start_time = time.time()
    results = []
    for profile in test_profiles:
        result = classifier.predict_profile(profile)
        results.append(result)
    end_time = time.time()
    
    total_time = (end_time - start_time) * 1000
    avg_time = total_time / len(test_profiles)
    
    print(f"   Total Time: {total_time:.1f}ms")
    print(f"   Average Time: {avg_time:.1f}ms per profile")
    print(f"   Profiles per Second: {len(test_profiles) / (total_time / 1000):.1f}")
    
    # Show sample results
    print(f"   Sample Results:")
    for i, result in enumerate(results[:3]):
        print(f"     Profile {i+1}: Score={result['authenticity_score']:.3f}, Fake={result['probability']:.3f}")
    
    print("\n" + "=" * 50)
    print("✅ Direct Model Testing Completed!")

if __name__ == "__main__":
    test_model_directly() 