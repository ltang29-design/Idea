import pandas as pd
import numpy as np
from profile_authenticity_classifier import EnhancedProfileAuthenticityClassifier
import time
from datetime import datetime
import json

class ModelAccuracyTester:
    def __init__(self):
        self.classifier = EnhancedProfileAuthenticityClassifier()
        self.load_models()
        
    def load_models(self):
        """Load the trained models"""
        try:
            self.classifier.load_models()
            print("✅ Models loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            return False
        return True
    
    def test_on_training_data(self):
        """Test accuracy on the training dataset"""
        print("\n" + "="*60)
        print("🧪 TESTING ON TRAINING DATA")
        print("="*60)
        
        # Load training data
        df = pd.read_json("comprehensive_profiles.json", lines=True)
        print(f"📊 Training dataset: {len(df)} profiles")
        print(f"   Real profiles: {len(df[df['label'] == 1])}")
        print(f"   Fake profiles: {len(df[df['label'] == 0])}")
        
        # Test predictions
        correct_predictions = 0
        total_predictions = 0
        processing_times = []
        
        for idx, row in df.iterrows():
            start_time = time.time()
            
            # Convert to dict format
            profile_dict = {
                'name': row['name'],
                'email': row['email'],
                'address': row['address'],
                'job_history': row['job_history'],
                'education': row['education'],
                'photo_flag': row['photo_flag']
            }
            
            # Get prediction
            result = self.classifier.predict_profile(profile_dict)
            prediction = 1 if result['authenticity_score'] > 0.5 else 0
            actual = row['label']
            
            # Check if correct
            if prediction == actual:
                correct_predictions += 1
            total_predictions += 1
            
            processing_time = (time.time() - start_time) * 1000
            processing_times.append(processing_time)
            
            # Print progress
            if (idx + 1) % 100 == 0:
                print(f"   Processed {idx + 1}/{len(df)} profiles...")
        
        accuracy = correct_predictions / total_predictions
        avg_processing_time = np.mean(processing_times)
        
        print(f"\n📈 Training Data Results:")
        print(f"   Accuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
        print(f"   Average Processing Time: {avg_processing_time:.2f}ms")
        print(f"   Total Processing Time: {sum(processing_times)/1000:.2f}s")
        
        return accuracy, avg_processing_time
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        print("\n" + "="*60)
        print("🔍 TESTING EDGE CASES")
        print("="*60)
        
        edge_cases = [
            # Obviously fake profiles
            {
                'name': 'James Bond',
                'email': 'james.bond@mi6.gov.uk',
                'address': '007 Secret Street, London, UK',
                'job_history': [
                    {'title': 'Secret Agent', 'company': 'MI6', 'start': 2000, 'end': 2024}
                ],
                'education': [
                    {'school': 'Spy Academy', 'degree': 'Secret Agent Diploma'}
                ],
                'photo_flag': 1,
                'expected': 0,
                'description': 'Obviously fake - fictional character'
            },
            # Very suspicious AI-generated content
            {
                'name': 'John Smith',
                'email': 'john.smith@gmail.com',
                'address': '123 Main St, New York, NY',
                'job_history': [
                    {'title': 'Exceptional Senior Software Engineer', 'company': 'Outstanding Tech Solutions Inc', 'start': 2018, 'end': 2024}
                ],
                'education': [
                    {'school': 'Phenomenal University', 'degree': 'Brilliant Computer Science Degree'}
                ],
                'photo_flag': 0,
                'expected': 0,
                'description': 'AI-generated overly positive language'
            },
            # Timeline overlaps
            {
                'name': 'Jane Doe',
                'email': 'jane.doe@yahoo.com',
                'address': '456 Oak Ave, San Francisco, CA',
                'job_history': [
                    {'title': 'Software Engineer', 'company': 'Google', 'start': 2015, 'end': 2020},
                    {'title': 'Senior Engineer', 'company': 'Microsoft', 'start': 2018, 'end': 2023}
                ],
                'education': [
                    {'school': 'Stanford University', 'degree': 'BSc Computer Science'}
                ],
                'photo_flag': 0,
                'expected': 0,
                'description': 'Employment timeline overlaps'
            },
            # Very realistic profile
            {
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
                'photo_flag': 0,
                'expected': 1,
                'description': 'Very realistic authentic profile'
            },
            # Suspicious email pattern
            {
                'name': 'Michael Brown',
                'email': 'michaelbrown123@gmail.com',
                'address': '321 Elm St, Boston, MA',
                'job_history': [
                    {'title': 'Data Scientist', 'company': 'Amazon', 'start': 2019, 'end': 2024}
                ],
                'education': [
                    {'school': 'MIT', 'degree': 'MSc Data Science'}
                ],
                'photo_flag': 0,
                'expected': 0,
                'description': 'Suspicious email pattern (no dots)'
            }
        ]
        
        correct_edge_cases = 0
        total_edge_cases = len(edge_cases)
        
        for i, case in enumerate(edge_cases, 1):
            print(f"\n🔍 Test Case {i}: {case['description']}")
            
            # Get prediction
            result = self.classifier.predict_profile(case)
            prediction = 1 if result['authenticity_score'] > 0.5 else 0
            expected = case['expected']
            
            # Check if correct
            is_correct = prediction == expected
            if is_correct:
                correct_edge_cases += 1
                status = "✅ CORRECT"
            else:
                status = "❌ WRONG"
            
            print(f"   Expected: {'Real' if expected == 1 else 'Fake'}")
            print(f"   Predicted: {'Real' if prediction == 1 else 'Fake'}")
            print(f"   Authenticity Score: {result['authenticity_score']:.3f}")
            print(f"   Fake Probability: {result['probability']:.3f}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Status: {status}")
            
            # Show individual model scores
            if 'individual_scores' in result:
                print(f"   Individual Scores:")
                for model, score in result['individual_scores'].items():
                    print(f"     {model}: {score:.3f}")
        
        edge_case_accuracy = correct_edge_cases / total_edge_cases
        print(f"\n📊 Edge Case Results:")
        print(f"   Accuracy: {edge_case_accuracy:.4f} ({correct_edge_cases}/{total_edge_cases})")
        
        return edge_case_accuracy
    
    def test_performance_benchmark(self):
        """Benchmark performance with multiple profiles"""
        print("\n" + "="*60)
        print("⚡ PERFORMANCE BENCHMARK")
        print("="*60)
        
        # Create test profiles
        test_profiles = []
        for i in range(100):
            profile = {
                'name': f'Test User {i}',
                'email': f'test.user{i}@gmail.com',
                'address': f'{i} Test St, Test City, TC',
                'job_history': [
                    {'title': 'Software Engineer', 'company': 'Test Company', 'start': 2020, 'end': 2024}
                ],
                'education': [
                    {'school': 'Test University', 'degree': 'BSc Computer Science'}
                ],
                'photo_flag': 0
            }
            test_profiles.append(profile)
        
        print(f"📊 Testing {len(test_profiles)} profiles...")
        
        start_time = time.time()
        processing_times = []
        
        for i, profile in enumerate(test_profiles):
            profile_start = time.time()
            result = self.classifier.predict_profile(profile)
            profile_time = (time.time() - profile_start) * 1000
            processing_times.append(profile_time)
            
            if (i + 1) % 20 == 0:
                print(f"   Processed {i + 1}/{len(test_profiles)} profiles...")
        
        total_time = time.time() - start_time
        avg_time = np.mean(processing_times)
        min_time = np.min(processing_times)
        max_time = np.max(processing_times)
        
        print(f"\n📈 Performance Results:")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Average Time: {avg_time:.2f}ms")
        print(f"   Min Time: {min_time:.2f}ms")
        print(f"   Max Time: {max_time:.2f}ms")
        print(f"   Profiles per Second: {len(test_profiles)/total_time:.1f}")
        
        return avg_time, total_time
    
    def test_model_consistency(self):
        """Test model consistency with repeated predictions"""
        print("\n" + "="*60)
        print("🔄 MODEL CONSISTENCY TEST")
        print("="*60)
        
        # Test profile
        test_profile = {
            'name': 'Consistency Test User',
            'email': 'consistency.test@gmail.com',
            'address': '123 Consistency St, Test City, TC',
            'job_history': [
                {'title': 'Software Engineer', 'company': 'Test Company', 'start': 2020, 'end': 2024}
            ],
            'education': [
                {'school': 'Test University', 'degree': 'BSc Computer Science'}
            ],
            'photo_flag': 0
        }
        
        print(f"🔄 Running consistency test with 10 predictions...")
        
        results = []
        for i in range(10):
            result = self.classifier.predict_profile(test_profile)
            results.append({
                'authenticity_score': result['authenticity_score'],
                'probability': result['probability'],
                'confidence': result['confidence']
            })
            print(f"   Run {i+1}: Score={result['authenticity_score']:.4f}, Prob={result['probability']:.4f}, Conf={result['confidence']:.4f}")
        
        # Check consistency
        authenticity_scores = [r['authenticity_score'] for r in results]
        probabilities = [r['probability'] for r in results]
        confidences = [r['confidence'] for r in results]
        
        score_std = np.std(authenticity_scores)
        prob_std = np.std(probabilities)
        conf_std = np.std(confidences)
        
        print(f"\n📊 Consistency Results:")
        print(f"   Authenticity Score Std: {score_std:.6f}")
        print(f"   Probability Std: {prob_std:.6f}")
        print(f"   Confidence Std: {conf_std:.6f}")
        print(f"   Consistent: {'✅' if score_std < 0.001 else '❌'}")
        
        return score_std < 0.001
    
    def run_comprehensive_test(self):
        """Run all tests and generate comprehensive report"""
        print("🚀 COMPREHENSIVE MODEL ACCURACY TEST")
        print("="*60)
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        results = {}
        
        # Test 1: Training data accuracy
        results['training_accuracy'], results['avg_processing_time'] = self.test_on_training_data()
        
        # Test 2: Edge cases
        results['edge_case_accuracy'] = self.test_edge_cases()
        
        # Test 3: Performance benchmark
        results['avg_benchmark_time'], results['total_benchmark_time'] = self.test_performance_benchmark()
        
        # Test 4: Model consistency
        results['model_consistent'] = self.test_model_consistency()
        
        # Generate final report
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        print(f"🎯 Overall Results:")
        print(f"   Training Data Accuracy: {results['training_accuracy']:.4f}")
        print(f"   Edge Case Accuracy: {results['edge_case_accuracy']:.4f}")
        print(f"   Average Processing Time: {results['avg_processing_time']:.2f}ms")
        print(f"   Benchmark Processing Time: {results['avg_benchmark_time']:.2f}ms")
        print(f"   Model Consistency: {'✅ Consistent' if results['model_consistent'] else '❌ Inconsistent'}")
        
        # Overall assessment
        overall_score = (results['training_accuracy'] + results['edge_case_accuracy']) / 2
        print(f"\n🏆 Overall Model Score: {overall_score:.4f}")
        
        if overall_score >= 0.95:
            assessment = "🟢 EXCELLENT"
        elif overall_score >= 0.90:
            assessment = "🟡 GOOD"
        elif overall_score >= 0.80:
            assessment = "🟠 FAIR"
        else:
            assessment = "🔴 POOR"
        
        print(f"📊 Assessment: {assessment}")
        
        # Save results
        results['timestamp'] = datetime.now().isoformat()
        results['overall_score'] = overall_score
        results['assessment'] = assessment
        
        # Convert boolean to string for JSON serialization
        results['model_consistent'] = str(results['model_consistent'])
        
        with open('model_accuracy_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: model_accuracy_results.json")
        print(f"⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return results

if __name__ == "__main__":
    tester = ModelAccuracyTester()
    results = tester.run_comprehensive_test() 