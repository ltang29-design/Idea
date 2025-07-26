import pandas as pd
import numpy as np
from profile_authenticity_classifier import EnhancedProfileAuthenticityClassifier
import joblib

def test_model_loading():
    """Test if models are loading correctly"""
    print("🔍 Testing Model Loading...")
    
    classifier = EnhancedProfileAuthenticityClassifier()
    
    try:
        classifier.load_models()
        print("✅ Models loaded successfully")
        
        # Check if models exist
        print(f"   Ensemble model exists: {hasattr(classifier, 'ensemble') and classifier.ensemble is not None}")
        print(f"   Feature selector exists: {hasattr(classifier, 'feature_selector') and classifier.feature_selector is not None}")
        print(f"   Scaler exists: {hasattr(classifier, 'scaler') and classifier.scaler is not None}")
        
        return True
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        return False

def test_simple_predictions():
    """Test simple predictions with known profiles"""
    print("\n🔍 Testing Simple Predictions...")
    
    classifier = EnhancedProfileAuthenticityClassifier()
    classifier.load_models()
    
    # Test cases
    test_cases = [
        {
            'name': 'Obviously Fake',
            'profile': {
                'name': 'James Bond',
                'email': 'james.bond@mi6.gov.uk',
                'address': '007 Secret Street, London, UK',
                'job_history': [
                    {'title': 'Secret Agent', 'company': 'MI6', 'start': 2000, 'end': 2024}
                ],
                'education': [
                    {'school': 'Spy Academy', 'degree': 'Secret Agent Diploma'}
                ],
                'photo_flag': 1
            },
            'expected': 'fake'
        },
        {
            'name': 'Realistic Profile',
            'profile': {
                'name': 'Emily Chen',
                'email': 'emily.chen@gmail.com',
                'address': '789 Pine Rd, Chicago, IL',
                'job_history': [
                    {'title': 'Software Engineer', 'company': 'Google', 'start': 2018, 'end': 2021},
                    {'title': 'Senior Software Engineer', 'company': 'Microsoft', 'start': 2021, 'end': 2024}
                ],
                'education': [
                    {'school': 'UC Berkeley', 'degree': 'BSc Computer Science'}
                ],
                'photo_flag': 0
            },
            'expected': 'real'
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {case['name']}")
        
        try:
            result = classifier.predict_profile(case['profile'])
            
            prediction = 'real' if result['authenticity_score'] > 0.5 else 'fake'
            is_correct = prediction == case['expected']
            
            print(f"   Expected: {case['expected']}")
            print(f"   Predicted: {prediction}")
            print(f"   Authenticity Score: {result['authenticity_score']:.3f}")
            print(f"   Fake Probability: {result['probability']:.3f}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Status: {'✅ CORRECT' if is_correct else '❌ WRONG'}")
            
            if 'individual_scores' in result:
                print(f"   Individual Scores:")
                for model, score in result['individual_scores'].items():
                    print(f"     {model}: {score:.3f}")
                    
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_training_data_sample():
    """Test on a small sample of training data"""
    print("\n🔍 Testing Training Data Sample...")
    
    # Load a small sample of training data
    df = pd.read_json("comprehensive_profiles.json", lines=True)
    sample_df = df.head(10)  # Test first 10 profiles
    
    print(f"📊 Testing {len(sample_df)} profiles from training data")
    
    classifier = EnhancedProfileAuthenticityClassifier()
    classifier.load_models()
    
    correct = 0
    total = 0
    
    for idx, row in sample_df.iterrows():
        profile_dict = {
            'name': row['name'],
            'email': row['email'],
            'address': row['address'],
            'job_history': row['job_history'],
            'education': row['education'],
            'photo_flag': row['photo_flag']
        }
        
        try:
            result = classifier.predict_profile(profile_dict)
            prediction = 1 if result['authenticity_score'] > 0.5 else 0
            actual = row['label']
            
            is_correct = prediction == actual
            if is_correct:
                correct += 1
            total += 1
            
            print(f"   Profile {idx+1}: {row['name']}")
            print(f"     Actual: {'Real' if actual == 1 else 'Fake'}")
            print(f"     Predicted: {'Real' if prediction == 1 else 'Fake'}")
            print(f"     Score: {result['authenticity_score']:.3f}")
            print(f"     Status: {'✅' if is_correct else '❌'}")
            
        except Exception as e:
            print(f"   Profile {idx+1}: Error - {e}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"\n📈 Sample Accuracy: {accuracy:.4f} ({correct}/{total})")

def check_model_files():
    """Check if model files exist"""
    print("\n🔍 Checking Model Files...")
    
    import os
    
    model_files = [
        'enhanced_model_ensemble.pkl',
        'enhanced_model_scaler.pkl',
        'enhanced_model_selector.pkl',
        'enhanced_model_params.json'
    ]
    
    for file in model_files:
        exists = os.path.exists(file)
        size = os.path.getsize(file) if exists else 0
        print(f"   {file}: {'✅' if exists else '❌'} ({size} bytes)")

if __name__ == "__main__":
    print("🚀 Simple Model Accuracy Test")
    print("="*50)
    
    # Check model files
    check_model_files()
    
    # Test model loading
    if test_model_loading():
        # Test simple predictions
        test_simple_predictions()
        
        # Test training data sample
        test_training_data_sample()
    else:
        print("❌ Cannot proceed without loaded models") 