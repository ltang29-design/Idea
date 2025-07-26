import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestClassifier, IsolationForest, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import (
    StratifiedKFold, GridSearchCV, cross_val_score, train_test_split
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    classification_report
)
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from textblob import TextBlob
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnhancedProfileAuthenticityClassifier:
    def __init__(self):
        # Initialize NLP models
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Fake templates for detection
        self.fake_templates = [
            "Experienced professional seeking challenging role at your company.",
            "Results-driven individual with a proven track record.",
            "Dynamic leader with extensive experience in industry."
        ]
        self.fake_template_embeds = self.model.encode(self.fake_templates)
        
        # Initialize models
        self.models = {}
        self.best_params = {}
        self.feature_selector = None
        self.scaler = RobustScaler()
        
    def extract_enhanced_features(self, df):
        """Extract comprehensive features with better engineering"""
        print(f"🔍 Extracting features for {len(df)} profiles...")
        
        features = []
        
        # TF-IDF for job descriptions
        tfidf = TfidfVectorizer(max_features=100, stop_words='english')
        all_job_descs = [" ".join([j["title"] + " at " + j["company"] for j in jobs]) for jobs in df["job_history"]]
        tfidf_matrix = tfidf.fit_transform(all_job_descs)
        
        for idx, row in df.iterrows():
            feature_dict = {}
            
            # 1. TF-IDF Features
            tfidf_feat = tfidf_matrix[idx].toarray().flatten()
            feature_dict.update({
                'tfidf_mean': np.mean(tfidf_feat),
                'tfidf_std': np.std(tfidf_feat),
                'tfidf_max': np.max(tfidf_feat),
                'tfidf_nonzero': np.count_nonzero(tfidf_feat)
            })
            
            # 2. Template Similarity
            job_desc = " ".join([j["title"] + " at " + j["company"] for j in row["job_history"]])
            job_embed = self.model.encode(job_desc)
            similarities = util.cos_sim(job_embed, self.fake_template_embeds).numpy().flatten()
            feature_dict.update({
                'template_sim_max': np.max(similarities),
                'template_sim_mean': np.mean(similarities),
                'template_sim_std': np.std(similarities)
            })
            
            # 3. NER Features
            if self.nlp:
                doc = self.nlp(job_desc)
                orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
                gpes = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
                feature_dict.update({
                    'org_count': len(orgs),
                    'gpe_count': len(gpes),
                    'total_entities': len(orgs) + len(gpes)
                })
            else:
                feature_dict.update({
                    'org_count': 0, 'gpe_count': 0, 'total_entities': 0
                })
            
            # 4. Timeline Consistency
            jobs = sorted(row["job_history"], key=lambda x: x["start"])
            overlaps = 0
            for i in range(len(jobs) - 1):
                if jobs[i]["end"] > jobs[i + 1]["start"]:
                    overlaps += 1
            feature_dict.update({
                'timeline_overlaps': overlaps,
                'timeline_consistency': 1 - overlaps / max(len(jobs) - 1, 1)
            })
            
            # 5. Contact Analysis
            email = row["email"]
            address = row["address"]
            email_suspicious = int("." not in email.split("@")[0])
            address_suspicious = int("nowhere" in address.lower() or "fake" in address.lower())
            feature_dict.update({
                'email_suspicious': email_suspicious,
                'address_suspicious': address_suspicious,
                'contact_suspicious': email_suspicious + address_suspicious
            })
            
            # 6. Cultural Analysis
            name = row["name"]
            education = row["education"]
            western_names = ['John', 'Emily', 'Michael', 'Sophie', 'David', 'Anna']
            asian_universities = ['Tsinghua', 'Peking', 'Fudan']
            
            name_first = name.split()[0]
            has_western_name = name_first in western_names
            has_asian_uni = any(any(asian in edu["school"] for asian in asian_universities) for edu in education)
            
            feature_dict.update({
                'cultural_mismatch': int(has_western_name and has_asian_uni),
                'western_name': int(has_western_name),
                'asian_education': int(has_asian_uni)
            })
            
            # 7. Completeness Analysis
            generic_companies = ['Company Name', 'ABC Corp']
            generic_schools = ['University Name', 'College Name']
            
            generic_company_count = sum(1 for job in jobs if any(gen in job["company"] for gen in generic_companies))
            generic_school_count = sum(1 for edu in education if any(gen in edu["school"] for gen in generic_schools))
            
            feature_dict.update({
                'generic_company_ratio': generic_company_count / max(len(jobs), 1),
                'generic_school_ratio': generic_school_count / max(len(education), 1),
                'profile_completeness': 1 - (generic_company_count + generic_school_count) / max(len(jobs) + len(education), 1)
            })
            
            # 8. Anomaly Analysis
            suspicious_titles = ['ninja', 'guru', 'rockstar', 'hacker']
            suspicious_degrees = ['diploma', 'certificate']
            
            suspicious_title_count = sum(1 for job in jobs if any(susp in job["title"].lower() for susp in suspicious_titles))
            suspicious_degree_count = sum(1 for edu in education if any(susp in edu["degree"].lower() for susp in suspicious_degrees))
            
            feature_dict.update({
                'suspicious_titles': suspicious_title_count,
                'suspicious_degrees': suspicious_degree_count,
                'photo_flag': row.get("photo_flag", 0),
                'total_suspicious': suspicious_title_count + suspicious_degree_count + row.get("photo_flag", 0)
            })
            
            # 9. Sentiment Analysis
            combined_text = " ".join([
                name, address,
                " ".join([j["title"] + " " + j["company"] for j in jobs]),
                " ".join([edu["school"] + " " + edu["degree"] for edu in education])
            ])
            
            try:
                blob = TextBlob(combined_text)
                sentiment_polarity = blob.sentiment.polarity
                sentiment_subjectivity = blob.sentiment.subjectivity
            except:
                sentiment_polarity = 0.0
                sentiment_subjectivity = 0.0
            
            # AI-generated content indicators
            overly_positive = int(sentiment_polarity > 0.7)
            high_subjectivity = int(sentiment_subjectivity > 0.8)
            
            ai_phrases = ['excellent', 'outstanding', 'exceptional', 'remarkable', 'extraordinary']
            ai_phrase_count = sum(1 for phrase in ai_phrases if phrase.lower() in combined_text.lower())
            
            feature_dict.update({
                'sentiment_polarity': sentiment_polarity,
                'sentiment_subjectivity': sentiment_subjectivity,
                'overly_positive': overly_positive,
                'high_subjectivity': high_subjectivity,
                'ai_phrase_count': ai_phrase_count,
                'sentiment_suspicious': overly_positive + high_subjectivity + int(ai_phrase_count > 2)
            })
            
            features.append(feature_dict)
            
            if idx % 100 == 0:
                print(f"   Processed {idx + 1}/{len(df)} profiles")
        
        print(f"✅ Extracted {len(features)} feature dictionaries")
        feature_df = pd.DataFrame(features)
        print(f"📊 Feature matrix shape: {feature_df.shape}")
        return feature_df
    
    def train_with_cv(self, df, cv_folds=5):
        """Train model with comprehensive cross-validation and hyperparameter tuning"""
        print("🔍 Extracting enhanced features...")
        X = self.extract_enhanced_features(df)
        y = df["label"]
        
        print(f"📈 Class distribution: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Feature selection
        print("🎯 Performing feature selection...")
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(20, X.shape[1]))
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        X_test_selected = self.feature_selector.transform(X_test)
        
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        print(f"✅ Selected {len(selected_features)} features")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Initialize models
        models = {
            'logistic': LogisticRegression(random_state=42, max_iter=1000),
            'rf': RandomForestClassifier(random_state=42, n_estimators=100),
            'svm': SVC(random_state=42, probability=True),
            'isolation': IsolationForest(contamination=0.1, random_state=42)
        }
        
        # Hyperparameter grids
        param_grids = {
            'logistic': {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']},
            'rf': {'n_estimators': [50, 100], 'max_depth': [5, 10, None]},
            'svm': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']},
            'isolation': {'contamination': [0.05, 0.1, 0.15]}
        }
        
        # Train models
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        best_models = {}
        
        for name, model in models.items():
            print(f"\n🔧 Tuning {name.upper()} model...")
            
            if name == 'isolation':
                # IsolationForest doesn't support ROC-AUC
                grid_search = GridSearchCV(
                    model, param_grids[name], cv=cv, scoring='accuracy',
                    n_jobs=-1, verbose=0
                )
                grid_search.fit(X_train_scaled, y_train)
                
                best_models[name] = grid_search.best_estimator_
                self.best_params[name] = grid_search.best_params_
                
                anomaly_scores = grid_search.best_estimator_.decision_function(X_train_scaled)
                y_pred_anomaly = (anomaly_scores < 0).astype(int)
                accuracy = np.mean(y_pred_anomaly == y_train)
                
                print(f"✅ {name.upper()} - Best params: {grid_search.best_params_}")
                print(f"📊 Anomaly Detection Accuracy: {accuracy:.3f}")
            else:
                grid_search = GridSearchCV(
                    model, param_grids[name], cv=cv, scoring='roc_auc',
                    n_jobs=-1, verbose=0
                )
                grid_search.fit(X_train_scaled, y_train)
                
                best_models[name] = grid_search.best_estimator_
                self.best_params[name] = grid_search.best_params_
                
                cv_scores = cross_val_score(
                    best_models[name], X_train_scaled, y_train, 
                    cv=cv, scoring='roc_auc'
                )
                
                print(f"✅ {name.upper()} - Best params: {grid_search.best_params_}")
                print(f"📊 CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Create ensemble
        print("\n🤝 Creating ensemble model...")
        ensemble = VotingClassifier(
            estimators=[
                ('logistic', best_models['logistic']),
                ('rf', best_models['rf']),
                ('svm', best_models['svm'])
            ],
            voting='soft'
        )
        ensemble.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_ensemble = ensemble.predict(X_test_scaled)
        y_prob_ensemble = ensemble.predict_proba(X_test_scaled)[:, 1]
        
        isolation_scores = best_models['isolation'].decision_function(X_test_scaled)
        y_pred_isolation = (isolation_scores < 0).astype(int)
        
        print("\n📈 Final Results:")
        print(f"Ensemble Accuracy: {accuracy_score(y_test, y_pred_ensemble):.3f}")
        print(f"Ensemble ROC-AUC: {roc_auc_score(y_test, y_prob_ensemble):.3f}")
        print(f"IsolationForest Accuracy: {accuracy_score(y_test, y_pred_isolation):.3f}")
        
        # Store models
        self.models = best_models
        self.ensemble = ensemble
        
        # Feature importance
        rf_model = best_models['rf']
        feature_importance = pd.DataFrame({
            'feature': selected_features,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🏆 Top 10 Most Important Features:")
        print(feature_importance.head(10))
        
        return {
            'ensemble_accuracy': accuracy_score(y_test, y_pred_ensemble),
            'ensemble_auc': roc_auc_score(y_test, y_prob_ensemble),
            'isolation_accuracy': accuracy_score(y_test, y_pred_isolation),
            'feature_importance': feature_importance,
            'best_params': self.best_params
        }
    
    def predict_profile(self, profile):
        """Predict authenticity with ensemble model and IsolationForest"""
        df = pd.DataFrame([profile])
        X = self.extract_enhanced_features(df)
        
        # Apply feature selection
        X_selected = self.feature_selector.transform(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X_selected)
        
        # Get ensemble prediction
        prob_authentic = self.ensemble.predict_proba(X_scaled)[0, 1]  # Probability of class 1 (authentic)
        prob_fake = 1 - prob_authentic  # Probability of being fake
        prediction = int(prob_fake > 0.5)  # 1 if fake, 0 if real
        
        # Get individual model predictions
        individual_predictions = {}
        for name, model in self.models.items():
            if name == 'isolation':
                anomaly_score = model.decision_function(X_scaled)[0]
                individual_predictions[name] = float(anomaly_score)
            else:
                # Convert authentic probability to fake probability
                prob_authentic = model.predict_proba(X_scaled)[0, 1]
                individual_predictions[name] = 1 - prob_authentic
        
        # Get IsolationForest anomaly score
        isolation_score = self.models['isolation'].decision_function(X_scaled)[0]
        is_anomaly = isolation_score < 0
        
        return {
            'prediction': prediction,
            'probability': float(prob_fake),
            'authenticity_score': float(1 - prob_fake),
            'individual_scores': individual_predictions,
            'isolation_anomaly': bool(is_anomaly),
            'isolation_score': float(isolation_score),
            'confidence': abs(prob_fake - 0.5) * 2
        }
    
    def save_models(self, filepath_prefix="enhanced_model"):
        """Save all models and components"""
        joblib.dump(self.ensemble, f"{filepath_prefix}_ensemble.joblib")
        joblib.dump(self.models, f"{filepath_prefix}_models.joblib")
        joblib.dump(self.feature_selector, f"{filepath_prefix}_selector.joblib")
        joblib.dump(self.scaler, f"{filepath_prefix}_scaler.joblib")
        joblib.dump(self.best_params, f"{filepath_prefix}_params.joblib")
        print(f"✅ Models saved with prefix: {filepath_prefix}")
    
    def load_models(self, filepath_prefix="enhanced_model"):
        """Load all models and components"""
        self.ensemble = joblib.load(f"{filepath_prefix}_ensemble.joblib")
        self.models = joblib.load(f"{filepath_prefix}_models.joblib")
        self.feature_selector = joblib.load(f"{filepath_prefix}_selector.joblib")
        self.scaler = joblib.load(f"{filepath_prefix}_scaler.joblib")
        self.best_params = joblib.load(f"{filepath_prefix}_params.joblib")
        print(f"✅ Models loaded from prefix: {filepath_prefix}")

if __name__ == "__main__":
    # Load data
    df = pd.read_json("comprehensive_profiles.json", lines=True)
    
    # Initialize enhanced classifier
    classifier = EnhancedProfileAuthenticityClassifier()
    
    # Train with CV
    results = classifier.train_with_cv(df, cv_folds=5)
    
    # Save models
    classifier.save_models()
    
    print("\n🎉 Enhanced model training completed!")
    print(f"📊 Final Ensemble AUC: {results['ensemble_auc']:.3f}")
    print(f"🔍 IsolationForest Accuracy: {results['isolation_accuracy']:.3f}") 