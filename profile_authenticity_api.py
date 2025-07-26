from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from profile_authenticity_classifier import EnhancedProfileAuthenticityClassifier
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Enhanced Profile Authenticity Classifier API",
    description="AI-powered profile authenticity detection with ensemble models and cross-validation",
    version="2.0.0"
)

# Pydantic models
class Job(BaseModel):
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    start: int = Field(..., description="Start year")
    end: int = Field(..., description="End year")

class Education(BaseModel):
    school: str = Field(..., description="School/University name")
    degree: str = Field(..., description="Degree type")

class Profile(BaseModel):
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Physical address")
    job_history: List[Job] = Field(..., description="Employment history")
    education: List[Education] = Field(..., description="Educational background")
    photo_flag: int = Field(0, description="Photo authenticity flag (0=real, 1=suspicious)")

class BatchProfileRequest(BaseModel):
    profiles: List[Profile] = Field(..., description="List of profiles to validate")

class ProfileResponse(BaseModel):
    profile_id: str
    authenticity_score: float
    fake_probability: float
    risk_level: str
    confidence: float
    individual_scores: Dict[str, float]
    isolation_anomaly: bool
    isolation_score: float
    flags: List[str]
    processing_time_ms: float
    timestamp: str

class BatchResponse(BaseModel):
    results: List[ProfileResponse]
    summary: Dict[str, float]
    processing_time_ms: float
    timestamp: str

# Initialize classifier
classifier = EnhancedProfileAuthenticityClassifier()

def load_models():
    """Load trained models"""
    try:
        classifier.load_models()
        logger.info("✅ Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("🚀 Starting Enhanced Profile Authenticity Classifier API...")
    if not load_models():
        logger.warning("⚠️ Models not loaded. Please train models first.")

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    models_loaded = hasattr(classifier, 'ensemble') and classifier.ensemble is not None
    return {
        "status": "healthy" if models_loaded else "models_not_loaded",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "models_loaded": models_loaded,
        "available_endpoints": [
            "POST /validate-profile",
            "POST /batch-validate",
            "GET /health",
            "GET /model-info"
        ]
    }

@app.get("/model-info", tags=["Model"])
async def model_info():
    """Get information about the trained models"""
    if not hasattr(classifier, 'ensemble') or classifier.ensemble is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "model_type": "Ensemble (Logistic Regression + Random Forest + SVM)",
        "feature_selection": "SelectKBest with f_classif",
        "scaling": "RobustScaler",
        "cross_validation": "StratifiedKFold (5 folds)",
        "hyperparameter_tuning": "GridSearchCV",
        "best_parameters": classifier.best_params,
        "feature_count": classifier.feature_selector.k_ if classifier.feature_selector else "Unknown",
        "training_timestamp": "Available in model files"
    }

@app.post("/validate-profile", response_model=ProfileResponse, tags=["Validation"])
async def validate_profile(profile: Profile, background_tasks: BackgroundTasks):
    """Validate a single profile for authenticity"""
    start_time = datetime.now()
    
    try:
        if not hasattr(classifier, 'ensemble') or classifier.ensemble is None:
            raise HTTPException(status_code=503, detail="Models not loaded. Please train models first.")
        
        # Convert profile to dict
        profile_dict = profile.dict()
        
        # Predict authenticity
        result = classifier.predict_profile(profile_dict)
        
        # Generate risk level
        prob_fake = result['probability']
        if prob_fake > 0.8:
            risk_level = "high"
        elif prob_fake > 0.5:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Generate flags
        flags = generate_flags(profile_dict, result)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response = ProfileResponse(
            profile_id=profile_dict.get('name', 'unknown'),
            authenticity_score=result['authenticity_score'],
            fake_probability=result['probability'],
            risk_level=risk_level,
            confidence=result['confidence'],
            individual_scores=result['individual_scores'],
            isolation_anomaly=result['isolation_anomaly'],
            isolation_score=result['isolation_score'],
            flags=flags,
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
        # Log prediction
        logger.info(f"Profile validation: {profile.name} - Risk: {risk_level} - Score: {result['authenticity_score']:.3f}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error validating profile: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.post("/batch-validate", response_model=BatchResponse, tags=["Validation"])
async def batch_validate(request: BatchProfileRequest, background_tasks: BackgroundTasks):
    """Validate multiple profiles in batch"""
    start_time = datetime.now()
    
    try:
        if not hasattr(classifier, 'ensemble') or classifier.ensemble is None:
            raise HTTPException(status_code=503, detail="Models not loaded. Please train models first.")
        
        if len(request.profiles) > 100:
            raise HTTPException(status_code=400, detail="Batch size too large. Maximum 100 profiles allowed.")
        
        results = []
        for profile in request.profiles:
            profile_dict = profile.dict()
            result = classifier.predict_profile(profile_dict)
            
            # Generate risk level
            prob_fake = result['probability']
            if prob_fake > 0.8:
                risk_level = "high"
            elif prob_fake > 0.5:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            # Generate flags
            flags = generate_flags(profile_dict, result)
            
            profile_response = ProfileResponse(
                profile_id=profile_dict.get('name', 'unknown'),
                authenticity_score=result['authenticity_score'],
                fake_probability=result['probability'],
                risk_level=risk_level,
                confidence=result['confidence'],
                individual_scores=result['individual_scores'],
                isolation_anomaly=result['isolation_anomaly'],
                isolation_score=result['isolation_score'],
                flags=flags,
                processing_time_ms=0,  # Will be calculated for batch
                timestamp=datetime.now().isoformat()
            )
            results.append(profile_response)
        
        # Calculate summary statistics
        authenticity_scores = [r.authenticity_score for r in results]
        fake_probabilities = [r.fake_probability for r in results]
        risk_levels = [r.risk_level for r in results]
        
        summary = {
            "total_profiles": len(results),
            "avg_authenticity_score": np.mean(authenticity_scores),
            "avg_fake_probability": np.mean(fake_probabilities),
            "high_risk_count": risk_levels.count("high"),
            "medium_risk_count": risk_levels.count("medium"),
            "low_risk_count": risk_levels.count("low"),
            "high_risk_percentage": risk_levels.count("high") / len(results) * 100
        }
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"Batch validation completed: {len(results)} profiles processed in {processing_time:.2f}ms")
        
        return BatchResponse(
            results=results,
            summary=summary,
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in batch validation: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

def generate_flags(profile_dict: dict, result: dict) -> List[str]:
    """Generate human-readable flags for suspicious patterns"""
    flags = []
    
    # Email pattern flags
    email = profile_dict.get('email', '')
    if '.' not in email.split('@')[0]:
        flags.append("Email doesn't follow name.surname pattern")
    
    # Address flags
    address = profile_dict.get('address', '').lower()
    fake_indicators = ['nowhere', 'fake', 'unreal', 'imaginary', 'infinite loop']
    if any(indicator in address for indicator in fake_indicators):
        flags.append("Suspicious address detected")
    
    # Job history flags
    jobs = profile_dict.get('job_history', [])
    if len(jobs) > 1:
        # Check for overlaps
        sorted_jobs = sorted(jobs, key=lambda x: x['start'])
        for i in range(len(sorted_jobs) - 1):
            if sorted_jobs[i]['end'] > sorted_jobs[i + 1]['start']:
                flags.append("Employment timeline overlaps detected")
                break
    
    # Education flags
    education = profile_dict.get('education', [])
    for edu in education:
        if 'diploma' in edu['degree'].lower() or 'certificate' in edu['degree'].lower():
            flags.append("Suspicious degree type detected")
    
    # Sentiment analysis flags
    if 'individual_scores' in result:
        # Check for overly positive sentiment (AI-generated content indicator)
        if result.get('individual_scores', {}).get('sentiment_polarity', 0) > 0.7:
            flags.append("Overly positive sentiment detected (possible AI-generated content)")
        
        # Check for high subjectivity
        if result.get('individual_scores', {}).get('sentiment_subjectivity', 0) > 0.8:
            flags.append("High subjectivity detected (possible AI-generated content)")
    
    # IsolationForest anomaly flag
    if result.get('isolation_anomaly', False):
        flags.append("Anomaly detected by IsolationForest")
    
    # High fake probability flag
    if result['probability'] > 0.8:
        flags.append("Very high fake probability score")
    elif result['probability'] > 0.6:
        flags.append("High fake probability score")
    
    # Low confidence flag
    if result['confidence'] < 0.3:
        flags.append("Low confidence prediction")
    
    return flags

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced Profile Authenticity Classifier API",
        "version": "2.0.0",
        "description": "AI-powered profile authenticity detection with ensemble models",
        "documentation": "/docs",
        "health_check": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 