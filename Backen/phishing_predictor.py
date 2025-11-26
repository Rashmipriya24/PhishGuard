import pickle
import json
import pandas as pd
import re


class URLFeatureExtractor:
    def __init__(self):
        # Load your existing URL features
        with open('url_features.json', 'r') as f:
            self.feature_names = json.load(f)

        # Load your existing URL feature info
        with open('url_feature_info.json', 'r') as f:
            self.feature_info = json.load(f)

    def extract(self, url):
        """Extract features from URL using your existing logic"""
        url_str = str(url)
        features = {}

        # Extract features based on your trained model's expectations
        features['url_length'] = len(url_str)
        features['num_dots'] = url_str.count('.')
        features['num_hyphens'] = url_str.count('-')
        features['num_digits'] = sum(c.isdigit() for c in url_str)
        features['has_https'] = 1 if url_str.startswith('https') else 0

        # Suspicious keywords (adjust based on your training)
        suspicious_keywords = ['login', 'verify', 'security', 'account', 'password', 'bank', 'payment']
        features['suspicious_words'] = sum(1 for word in suspicious_keywords if word in url_str.lower())

        return features


class EmailFeatureExtractor:
    def __init__(self):
        # Load your existing email features
        with open('email_features.json', 'r') as f:
            self.feature_names = json.load(f)

        # Load your existing email feature info
        with open('email_feature_info.json', 'r') as f:
            self.feature_info = json.load(f)

    def extract(self, email_text):
        """Extract features from email using your existing logic"""
        text_str = str(email_text)
        features = {}

        # Extract features based on your trained model's expectations
        features['text_length'] = len(text_str)
        features['num_urgent_words'] = sum(
            1 for word in ['urgent', 'immediately', 'asap', 'alert'] if word in text_str.lower())
        features['num_special_chars'] = sum(1 for c in text_str if not c.isalnum() and not c.isspace())
        features['num_exclamation'] = text_str.count('!')
        features['has_winner_keywords'] = 1 if any(
            word in text_str.lower() for word in ['winner', 'prize', 'reward', 'won']) else 0

        return features


class PhishingPredictor:
    def __init__(self):
        # Load your existing trained models
        with open('url_model.pkl', 'rb') as f:
            self.url_model = pickle.load(f)

        with open('email_model.pkl', 'rb') as f:
            self.email_model = pickle.load(f)

        self.url_extractor = URLFeatureExtractor()
        self.email_extractor = EmailFeatureExtractor()

        print("✅ Phishing Predictor loaded with your trained models!")

    def predict_url(self, url):
        """Predict if URL is phishing - returns format frontend expects"""
        try:
            # Extract features using your existing logic
            features = self.url_extractor.extract(url)

            # Create feature vector in correct order
            feature_vector = [features.get(feature_name, 0) for feature_name in self.url_extractor.feature_names]

            # Use your trained model to predict
            prediction = self.url_model.predict([feature_vector])[0]
            probability = self.url_model.predict_proba([feature_vector])[0]

            # Generate explanation
            explanation = self._generate_url_explanation(features, prediction)

            # Return exactly what frontend expects
            return {
                "classification": "malicious" if prediction == 1 else "safe",
                "confidence": float(probability[1]) if prediction == 1 else float(probability[0]),
                "highlighted_text": self._highlight_suspicious_url(url),
                "explanation": explanation
            }

        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

    def predict_email(self, email_text):
        """Predict if email is phishing - returns format frontend expects"""
        try:
            # Extract features using your existing logic
            features = self.email_extractor.extract(email_text)

            # Create feature vector in correct order
            feature_vector = [features.get(feature_name, 0) for feature_name in self.email_extractor.feature_names]

            # Use your trained model to predict
            prediction = self.email_model.predict([feature_vector])[0]
            probability = self.email_model.predict_proba([feature_vector])[0]

            # Generate explanation
            explanation = self._generate_email_explanation(features, prediction)

            # Return exactly what frontend expects
            return {
                "classification": "malicious" if prediction == 1 else "safe",
                "confidence": float(probability[1]) if prediction == 1 else float(probability[0]),
                "highlighted_text": self._highlight_suspicious_email(email_text),
                "explanation": explanation
            }

        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

    def _generate_url_explanation(self, features, prediction):
        """Generate human-readable explanation for URL"""
        if prediction == 1:
            reasons = []
            if features.get('suspicious_words', 0) > 1:
                reasons.append("contains suspicious keywords")
            if features.get('num_hyphens', 0) > 2:
                reasons.append("has unusual domain structure")
            if features.get('has_https', 0) == 0:
                reasons.append("uses insecure HTTP protocol")

            if reasons:
                return f"URL classified as malicious because it {', '.join(reasons)}."
            else:
                return "URL shows patterns consistent with phishing attempts."
        else:
            return "URL appears safe based on security analysis."

    def _generate_email_explanation(self, features, prediction):
        """Generate human-readable explanation for email"""
        if prediction == 1:
            reasons = []
            if features.get('num_urgent_words', 0) > 0:
                reasons.append("uses urgent language")
            if features.get('num_exclamation', 0) > 1:
                reasons.append("uses excessive punctuation")
            if features.get('has_winner_keywords', 0) == 1:
                reasons.append("contains prize/winner mentions")

            if reasons:
                return f"Email classified as malicious because it {', '.join(reasons)}."
            else:
                return "Email shows patterns consistent with phishing attempts."
        else:
            return "Email content appears legitimate."

    def _highlight_suspicious_url(self, url):
        """Highlight suspicious parts of URL"""
        suspicious_words = ['login', 'verify', 'security', 'account', 'password', 'bank']
        highlighted = url

        for word in suspicious_words:
            if word in url.lower():
                highlighted = highlighted.replace(word, f"**{word}**")

        return highlighted

    def _highlight_suspicious_email(self, email_text):
        """Highlight suspicious parts of email"""
        suspicious_words = ['urgent', 'immediately', 'winner', 'prize', 'verify', 'password', 'account']
        highlighted = email_text

        for word in suspicious_words:
            if word in email_text.lower():
                highlighted = highlighted.replace(word, f"**{word}**")

        return highlighted