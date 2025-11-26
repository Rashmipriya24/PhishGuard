# train_phishing_models.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import json
import os
import re

print("🚀 Starting Phishing Detection Model Training...")


def load_datasets():
    """Load URL and Email datasets"""
    print("\n📥 Loading datasets...")

    # Check if files exist with correct names
    if not os.path.exists('malicious_phish.csv'):
        print("❌ malicious_phish.csv not found! Please ensure the file is in the same directory.")
        return None, None

    if not os.path.exists('email_phishing_data.csv'):
        print("❌ email_phishing_data.csv not found! Please ensure the file is in the same directory.")
        return None, None

    try:
        # Load datasets
        url_df = pd.read_csv('malicious_phish.csv')
        email_df = pd.read_csv('email_phishing_data.csv')

        print(f"✅ URL dataset loaded: {url_df.shape}")
        print(f"✅ Email dataset loaded: {email_df.shape}")

        return url_df, email_df

    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return None, None


def explore_dataset(df, name):
    """Explore dataset structure"""
    print(f"\n🔍 {name} Dataset Analysis:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))

    # Check for missing values
    print(f"\nMissing values:")
    print(df.isnull().sum())

    return df


def extract_url_features(urls):
    """Extract features from URL strings"""
    features = []

    for url in urls:
        url_str = str(url).strip()

        # Basic URL features
        url_features = {
            'url_length': len(url_str),
            'num_dots': url_str.count('.'),
            'num_hyphens': url_str.count('-'),
            'num_underscore': url_str.count('_'),
            'num_slash': url_str.count('/'),
            'num_question': url_str.count('?'),
            'num_equal': url_str.count('='),
            'num_ampersand': url_str.count('&'),
            'num_at': url_str.count('@'),
            'num_digits': sum(c.isdigit() for c in url_str),
            'has_https': 1 if url_str.startswith('https') else 0,
            'has_http': 1 if url_str.startswith('http') else 0,
        }

        # Suspicious patterns
        suspicious_keywords = ['login', 'verify', 'security', 'account', 'update', 'confirm',
                               'password', 'bank', 'payment', 'alert', 'secure', 'validation',
                               'signin', 'authenticate', 'recover', 'suspend']
        url_features['suspicious_words'] = sum(1 for word in suspicious_keywords if word in url_str.lower())

        # Check for IP address
        ip_pattern = r'\d+\.\d+\.\d+\.\d+'
        url_features['has_ip'] = 1 if re.search(ip_pattern, url_str) else 0

        # Domain length (approximate)
        domain_part = url_str.split('/')[2] if '//' in url_str else url_str.split('/')[0]
        url_features['domain_length'] = len(domain_part)

        features.append(url_features)

    return pd.DataFrame(features)


def extract_email_features(email_texts):
    """Extract features from email text"""
    features = []

    for text in email_texts:
        text_str = str(text).strip()

        email_features = {
            'text_length': len(text_str),
            'num_words': len(text_str.split()),
            'num_sentences': text_str.count('.') + text_str.count('!') + text_str.count('?'),
            'num_special_chars': sum(1 for c in text_str if not c.isalnum() and not c.isspace()),
            'num_exclamation': text_str.count('!'),
            'num_question': text_str.count('?'),
            'num_dollar': text_str.count('$'),
            'num_uppercase': sum(1 for c in text_str if c.isupper()),
            'uppercase_ratio': sum(1 for c in text_str if c.isupper()) / max(1, len(text_str)),
        }

        # Content-based features
        urgency_words = ['urgent', 'immediately', 'asap', 'instant', 'right away', 'emergency']
        threat_words = ['suspend', 'verify', 'confirm', 'validate', 'action required', 'limited time']
        reward_words = ['winner', 'prize', 'reward', 'free', 'bonus', 'congratulations']
        security_words = ['password', 'login', 'account', 'security', 'credentials']

        email_features['has_urgency'] = 1 if any(word in text_str.lower() for word in urgency_words) else 0
        email_features['has_threat'] = 1 if any(word in text_str.lower() for word in threat_words) else 0
        email_features['has_reward'] = 1 if any(word in text_str.lower() for word in reward_words) else 0
        email_features['has_security'] = 1 if any(word in text_str.lower() for word in security_words) else 0

        features.append(email_features)

    return pd.DataFrame(features)


def train_url_model(url_df):
    """Train URL phishing detection model"""
    print("\n" + "=" * 50)
    print("🤖 TRAINING URL PHISHING MODEL")
    print("=" * 50)

    # Auto-detect URL and label columns
    url_column = None
    label_column = None

    # Common column names in URL datasets
    url_columns = ['url', 'URL', 'website', 'link', 'domain', 'website_url']
    label_columns = ['label', 'type', 'class', 'result', 'isMalicious', 'malicious', 'target']

    # Find URL column
    for col in url_df.columns:
        if any(keyword in col.lower() for keyword in [uc.lower() for uc in url_columns]):
            url_column = col
            break
    if not url_column:
        url_column = url_df.columns[0]
        print(f"📝 Using first column as URL: '{url_column}'")
    else:
        print(f"📝 URL column: '{url_column}'")

    # Find label column
    for col in url_df.columns:
        if any(keyword in col.lower() for keyword in [lc.lower() for lc in label_columns]):
            label_column = col
            break
    if not label_column:
        if len(url_df.columns) > 1:
            label_column = url_df.columns[1]
            print(f"🏷️ Using second column as label: '{label_column}'")
        else:
            print("❌ No label column found in URL dataset")
            return None
    else:
        print(f"🏷️ Label column: '{label_column}'")

    # Prepare data
    urls = url_df[url_column].fillna('').astype(str)
    labels = url_df[label_column]

    # Convert labels to binary (0 = legitimate, 1 = phishing)
    original_labels = labels.copy()
    if labels.dtype == 'object':
        label_mapping = {
            'malicious': 1, 'benign': 0, 'bad': 1, 'good': 0,
            'phishing': 1, 'legitimate': 0, 'spam': 1, 'ham': 0,
            '1': 1, '0': 0, 1: 1, 0: 0
        }
        labels = labels.map(label_mapping).fillna(0).astype(int)

    print(f"📊 Label distribution:")
    print(f"   Legitimate (0): {(labels == 0).sum()} samples")
    print(f"   Phishing (1): {(labels == 1).sum()} samples")

    # Extract features
    print("🛠️ Extracting URL features...")
    X = extract_url_features(urls)
    y = labels

    print(f"🔧 Extracted {X.shape[1]} features from URLs")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"📚 Training samples: {X_train.shape[0]}")
    print(f"🧪 Testing samples: {X_test.shape[0]}")

    # Train model
    print("🎯 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n📈 MODEL PERFORMANCE:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

    print(f"📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   True Negative: {cm[0, 0]}")
    print(f"   False Positive: {cm[0, 1]}")
    print(f"   False Negative: {cm[1, 0]}")
    print(f"   True Positive: {cm[1, 1]}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n🎯 Top 10 Most Important Features:")
    for i, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")

    # Save model and metadata
    print("\n💾 Saving model...")
    with open('url_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('url_features.json', 'w') as f:
        json.dump(list(X.columns), f)

    model_info = {
        'url_column': url_column,
        'label_column': label_column,
        'accuracy': accuracy,
        'feature_names': list(X.columns),
        'model_type': 'RandomForest',
        'training_samples': len(X_train)
    }

    with open('url_model_info.json', 'w') as f:
        json.dump(model_info, f)

    print("✅ URL model training completed!")
    return model, X.columns


def train_email_model(email_df):
    """Train email phishing detection model"""
    print("\n" + "=" * 50)
    print("🤖 TRAINING EMAIL PHISHING MODEL")
    print("=" * 50)

    # Auto-detect text and label columns
    text_column = None
    label_column = None

    # Common column names in email datasets
    text_columns = ['text', 'content', 'body', 'message', 'email', 'subject', 'email_text']
    label_columns = ['label', 'type', 'class', 'phishing', 'spam', 'target', 'is_phishing']

    # Find text column
    for col in email_df.columns:
        if any(keyword in col.lower() for keyword in [tc.lower() for tc in text_columns]):
            text_column = col
            break

    # Find label column
    for col in email_df.columns:
        if any(keyword in col.lower() for keyword in [lc.lower() for lc in label_columns]):
            label_column = col
            break

    if not label_column:
        print("❌ No label column found in email dataset")
        return None

    if text_column:
        print(f"📝 Text column: '{text_column}'")
    else:
        print("📝 Using all columns except label as features")

    print(f"🏷️ Label column: '{label_column}'")

    # Prepare data
    if text_column:
        texts = email_df[text_column].fillna('').astype(str)
        X = extract_email_features(texts)
    else:
        # Use all other columns as features
        feature_cols = [col for col in email_df.columns if col != label_column]
        X = email_df[feature_cols].fillna(0)

    labels = email_df[label_column]

    # Convert labels to binary
    if labels.dtype == 'object':
        label_mapping = {
            'phishing': 1, 'legitimate': 0, 'spam': 1, 'ham': 0,
            'malicious': 1, 'benign': 0, '1': 1, '0': 0, 1: 1, 0: 0
        }
        y = labels.map(label_mapping).fillna(0).astype(int)
    else:
        y = labels.astype(int)

    print(f"📊 Label distribution:")
    print(f"   Legitimate (0): {(y == 0).sum()} samples")
    print(f"   Phishing (1): {(y == 1).sum()} samples")

    print(f"🔧 Using {X.shape[1]} features")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"📚 Training samples: {X_train.shape[0]}")
    print(f"🧪 Testing samples: {X_test.shape[0]}")

    # Train model
    print("🎯 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\n📈 MODEL PERFORMANCE:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Phishing']))

    print(f"📊 Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"   True Negative: {cm[0, 0]}")
    print(f"   False Positive: {cm[0, 1]}")
    print(f"   False Negative: {cm[1, 0]}")
    print(f"   True Positive: {cm[1, 1]}")

    # Save model and metadata
    print("\n💾 Saving model...")
    with open('email_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('email_features.json', 'w') as f:
        json.dump(list(X.columns), f)

    model_info = {
        'label_column': label_column,
        'text_column': text_column,
        'accuracy': accuracy,
        'feature_names': list(X.columns),
        'model_type': 'RandomForest',
        'training_samples': len(X_train)
    }

    with open('email_model_info.json', 'w') as f:
        json.dump(model_info, f)

    print("✅ Email model training completed!")
    return model, X.columns


def main():
    """Main training function"""
    # Load datasets
    url_df, email_df = load_datasets()
    if url_df is None or email_df is None:
        return

    # Explore datasets
    print("\n" + "=" * 60)
    url_df = explore_dataset(url_df, "URL (malicious_phish.csv)")
    print("\n" + "=" * 60)
    email_df = explore_dataset(email_df, "EMAIL (email_phishing_data.csv)")
    print("=" * 60)

    # Train models
    url_model, url_features = train_url_model(url_df)
    email_model, email_features = train_email_model(email_df)

    # Summary
    print("\n" + "🎉" * 20)
    print("🎯 TRAINING SUMMARY")
    print("🎉" * 20)

    if url_model:
        print("✅ URL Model: TRAINED SUCCESSFULLY")
        print(f"   - Features: {len(url_features)}")
        print(f"   - Model: Random Forest")
        print(f"   - File: url_model.pkl")

    if email_model:
        print("✅ Email Model: TRAINED SUCCESSFULLY")
        print(f"   - Features: {len(email_features)}")
        print(f"   - Model: Random Forest")
        print(f"   - File: email_model.pkl")

    print("\n📁 Generated Files:")
    files = ['url_model.pkl', 'email_model.pkl', 'url_features.json',
             'email_features.json', 'url_model_info.json', 'email_model_info.json']
    for file in files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} (missing)")

    print("\n🚀 Next step: Run 'test_phishing_models.py' to test the models!")


if __name__ == "__main__":
    main()