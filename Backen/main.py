from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json
import re


# Request models
class URLRequest(BaseModel):
    url: str


class EmailRequest(BaseModel):
    emailText: str


# Create FastAPI app
app = FastAPI(
    title="PhishGuard API",
    description="Real-time Phishing Detection Backend",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🔍 Loading phishing detection models...")

# Load your existing models
with open('../models/url_model.pkl', 'rb') as f:
    url_model = pickle.load(f)

    with open('../models/email_model.pkl', 'rb') as f:
        email_model = pickle.load(f)

    with open('../models/url_features.json', 'r') as f:
        url_features = json.load(f)


    with open('../models/email_features.json', 'r') as f:
        email_features = json.load(f)


print("✅ All models loaded successfully!")


def extract_url_features(url):
    """Extract features from URL"""
    url_str = str(url)
    return {
        'url_length': len(url_str),
        'num_dots': url_str.count('.'),
        'num_hyphens': url_str.count('-'),
        'num_slash': url_str.count('/'),
        'num_question': url_str.count('?'),
        'num_equal': url_str.count('='),
        'num_at': url_str.count('@'),
        'num_digits': sum(c.isdigit() for c in url_str),
        'has_https': 1 if url_str.startswith('https') else 0,
        'has_http': 1 if url_str.startswith('http') else 0,
        'suspicious_words': sum(
            1 for word in ['login', 'verify', 'security', 'account', 'password'] if word in url_str.lower())
    }


def extract_email_features(email_text):
    """Extract features from email text"""
    text_str = str(email_text)
    return {
        'text_length': len(text_str),
        'num_words': len(text_str.split()),
        'num_special_chars': sum(1 for c in text_str if not c.isalnum() and not c.isspace()),
        'num_exclamation': text_str.count('!'),
        'num_question': text_str.count('?'),
        'num_dollar': text_str.count('$'),
        'num_uppercase': sum(1 for c in text_str if c.isupper()),
        'has_urgent': 1 if any(word in text_str.lower() for word in ['urgent', 'immediately', 'asap']) else 0,
        'has_password': 1 if any(word in text_str.lower() for word in ['password', 'login', 'verify']) else 0,
        'has_winner': 1 if any(word in text_str.lower() for word in ['winner', 'prize', 'reward']) else 0
    }


def highlight_suspicious_url(url):
    """Highlight suspicious parts in URL"""
    suspicious_words = ['login', 'verify', 'security', 'account', 'password']
    highlighted = url
    for word in suspicious_words:
        if word in url.lower():
            highlighted = highlighted.replace(word, f"**{word}**")
    return highlighted


def highlight_suspicious_email(email_text):
    """Highlight suspicious parts in email"""
    suspicious_words = ['urgent', 'immediately', 'winner', 'prize', 'verify', 'password']
    highlighted = email_text
    for word in suspicious_words:
        if word in email_text.lower():
            highlighted = highlighted.replace(word, f"**{word}**")
    return highlighted


def generate_explanation(features, prediction, type):
    """Generate explanation for the prediction"""
    if prediction == 1:  # Malicious
        if type == 'url':
            reasons = []
            if features.get('suspicious_words', 0) > 1:
                reasons.append("suspicious keywords")
            if features.get('num_hyphens', 0) > 2:
                reasons.append("unusual domain structure")
            if features.get('has_https', 0) == 0:
                reasons.append("insecure connection")
            return f"URL shows {', '.join(reasons)} commonly found in phishing attempts."
        else:  # email
            reasons = []
            if features.get('has_urgent', 0) == 1:
                reasons.append("urgency language")
            if features.get('has_winner', 0) == 1:
                reasons.append("prize/winner mentions")
            if features.get('num_exclamation', 0) > 1:
                reasons.append("excessive punctuation")
            return f"Email contains {', '.join(reasons)} typical of phishing campaigns."
    else:  # Safe
        if type == 'url':
            return "URL appears legitimate based on security analysis."
        else:
            return "Email content appears safe and legitimate."


@app.post("/api/analyze/url")
async def analyze_url(request: dict):
    """Analyze URL for phishing"""
    try:
        url = request.get("url", "")

        if not url:
            return {"error": "No URL provided"}

        features = extract_url_features(url)
        feature_vector = [features.get(f, 0) for f in url_features]

        prediction = url_model.predict([feature_vector])[0]
        probability = url_model.predict_proba([feature_vector])[0]

        return {
            "classification": "malicious" if prediction == 1 else "safe",
            "confidence": float(probability[1]) if prediction == 1 else float(probability[0]),
            "highlighted_text": highlight_suspicious_url(url),
            "explanation": generate_explanation(features, prediction, 'url')
        }

    except Exception as e:
        return {"error": f"Server error: {str(e)}"}


@app.post("/api/analyze/email")
async def analyze_email(request: dict):
    """Analyze email content for phishing"""
    try:
        email_text = request.get("emailText", "")

        if not email_text:
            return {"error": "No email text provided"}

        features = extract_email_features(email_text)
        feature_vector = [features.get(f, 0) for f in email_features]

        prediction = email_model.predict([feature_vector])[0]
        probability = email_model.predict_proba([feature_vector])[0]

        return {
            "classification": "malicious" if prediction == 1 else "safe",
            "confidence": float(probability[1]) if prediction == 1 else float(probability[0]),
            "highlighted_text": highlight_suspicious_email(email_text),
            "explanation": generate_explanation(features, prediction, 'email')
        }

    except Exception as e:
        return {"error": f"Server error: {str(e)}"}


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "PhishGuard Backend API",
        "version": "1.0.0"
    }


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "PhishGuard API is running. Visit /docs for API documentation."}


if __name__ == "__main__":
    import uvicorn

    print("🚀 PhishGuard Backend API Starting...")
    print("📡 Endpoints:")
    print("   POST /api/analyze/url - Analyze URL for phishing")
    print("   POST /api/analyze/email - Analyze email content")
    print("   GET  /api/health - Health check")
    print("   GET  /docs - Interactive API documentation")
    print("\n🔗 React frontend should connect to: http://localhost:8000")

    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=8000)