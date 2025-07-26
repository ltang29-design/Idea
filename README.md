# Enhanced AI-Powered Profile Authenticity Classifier

A sophisticated machine learning system for detecting fake profiles in recruiting platforms using ensemble models, sentiment analysis, and anomaly detection.

## 🚀 5-Minute Setup Instructions

### Prerequisites
- Python 3.8+ installed
- Git (optional, for cloning)

### Step 1: Download/Clone the Repository
```bash
# Option A: Clone the repository
git clone <your-repo-url>
cd <repo-folder>

# Option B: Download and extract the ZIP file
# Then navigate to the extracted folder
```

### Step 2: Install Dependencies (2 minutes)
```bash
# Install all required packages
pip install -r requirements.txt

# Install spaCy language model
python -m spacy download en_core_web_sm
```

### Step 3: Generate Training Data (1 minute)
```bash
# Generate comprehensive synthetic dataset for training
python comprehensive_data_generator.py
```

### Step 4: Train the Enhanced Model (2 minutes)
```bash
# Train the enhanced model with sentiment analysis and IsolationForest
python profile_authenticity_classifier.py
```

### Step 5: Start the API Server
```bash
# Start the enhanced API server
uvicorn profile_authenticity_api:app --reload --host 0.0.0.0 --port 8000
```

### Step 6: Test the API
- Open your browser and go to: `http://localhost:8000/docs`
- Or use the provided demo scripts: `enhanced_demo_script_v2.http`

## 🎯 What's New in v2.0

### Enhanced Features
- **Sentiment Analysis**: Detects AI-generated content through overly positive sentiment
- **IsolationForest Anomaly Detection**: Identifies unusual profile patterns
- **Advanced Cross-Validation**: 5-fold stratified CV with hyperparameter tuning
- **Ensemble Models**: Combines Logistic Regression, Random Forest, and SVM
- **Feature Selection**: Automatic selection of top 20 most important features
- **Comprehensive Flagging**: Detailed explanations for suspicious patterns

### New API Endpoints
- `POST /validate-profile`: Single profile validation with detailed analysis
- `POST /batch-validate`: Process up to 100 profiles at once
- `GET /model-info`: Information about trained models
- `GET /health`: System health check

## 📊 Model Performance

### Enhanced Features (31 total)
1. **TF-IDF Analysis**: Text-based feature extraction
2. **Template Similarity**: Detection of fake profile templates
3. **Named Entity Recognition**: Company and location validation
4. **Timeline Consistency**: Employment overlap detection
5. **Contact Pattern Analysis**: Email and address validation
6. **Cultural Pattern Detection**: Name/university mismatch
7. **Profile Completeness**: Generic information detection
8. **Anomaly Indicators**: Suspicious titles and degrees
9. **Sentiment Analysis**: AI-generated content detection

### Model Ensemble
- **Logistic Regression**: With L1/L2 regularization
- **Random Forest**: With hyperparameter tuning
- **Support Vector Machine**: With kernel optimization
- **IsolationForest**: Anomaly detection

### Performance Metrics
- **Ensemble Accuracy**: 100% (on comprehensive synthetic data)
- **Cross-Validation AUC**: 1.000
- **Processing Time**: <100ms per profile
- **Dataset Size**: 800 profiles (400 real, 400 fake)
- **Industries Covered**: Tech, Finance, Consulting, Healthcare, Retail
- **Name Diversity**: 124 names across 4 ethnic categories

## 🔧 API Usage Examples

### Single Profile Validation
```bash
curl -X POST "http://localhost:8000/validate-profile" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

### Response Format
```json
{
  "profile_id": "John Smith",
  "authenticity_score": 0.95,
  "fake_probability": 0.05,
  "risk_level": "low",
  "confidence": 0.90,
  "individual_scores": {
    "logistic": 0.02,
    "rf": 0.03,
    "svm": 0.01,
    "isolation": 0.15
  },
  "isolation_anomaly": false,
  "isolation_score": 0.15,
  "flags": [],
  "processing_time_ms": 45.2,
  "timestamp": "2024-01-01T12:00:00"
}
```

## 📁 Project Structure

```
├── comprehensive_data_generator.py    # Comprehensive synthetic data generation
├── basic_data_generator.py            # Basic synthetic data generation (legacy)
├── profile_authenticity_classifier.py # Enhanced ML model with sentiment analysis
├── profile_authenticity_api.py        # FastAPI server with new endpoints
├── enhanced_demo_script_v2.http       # Demo API calls
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── ENHANCED_TECHNICAL_SUMMARY.md      # Technical details
├── comprehensive_profiles.json        # Comprehensive training data (800 profiles)
└── profiles.json                      # Basic training data (400 profiles)
```

## 🚨 Important Notes

### Prototype Limitations
- **Synthetic Data**: Uses simulated profiles for training
- **Feature Simulation**: Some features (address validation, photo analysis) are simulated
- **Model Performance**: Results on synthetic data may not reflect real-world performance

### Production Considerations
- **Real Data**: Integrate with actual profile databases
- **External APIs**: Connect to address validation and photo analysis services
- **Security**: Add authentication, rate limiting, and input validation
- **Monitoring**: Implement logging, metrics, and alerting
- **Scaling**: Use containerization and load balancing

## 🔍 Troubleshooting

### Common Issues

**Python not found:**
```bash
# Use python3 or py instead
python3 -m pip install -r requirements.txt
# or
py -m pip install -r requirements.txt
```

**spaCy model not found:**
```bash
python -m spacy download en_core_web_sm
```

**Port already in use:**
```bash
# Use a different port
uvicorn enhanced_api:app --reload --port 8001
```

**Memory issues:**
```bash
# Reduce batch size in API calls
# Or use smaller feature set in model training
```

## 📈 Next Steps

### Immediate Enhancements
1. **Real Data Integration**: Connect to actual profile databases
2. **External APIs**: Address validation, photo analysis, social media verification
3. **Model Monitoring**: Drift detection and retraining pipelines
4. **A/B Testing**: Compare model versions

### Advanced Features
1. **Deep Learning**: Transformer-based text analysis
2. **Image Analysis**: Photo authenticity detection
3. **Graph Analysis**: Social network validation
4. **Multi-language Support**: International profile validation

## 📞 Support

For questions or issues:
1. Check the troubleshooting section above
2. Review the technical summary in `ENHANCED_TECHNICAL_SUMMARY.md`
3. Test with the provided demo scripts
4. Check the API documentation at `http://localhost:8000/docs`

---

**Version**: 2.0.0  
**Last Updated**: January 2024  
**License**: MIT 