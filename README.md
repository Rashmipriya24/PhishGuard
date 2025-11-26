<div align="center">

# 🛡️ PhishGuard

### AI-Powered Real-Time Phishing Detection System

[![React](https://img.shields.io/badge/React-19.2-61DAFB?style=flat-square&logo=react)](https://reactjs.org/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**Detect phishing URLs and emails in real-time using machine learning**

[Features](#-features) • [Quick Start](#-quick-start) • [API](#-api) • [Tech Stack](#-tech-stack)

</div>

---

## ✨ Features

- 🤖 **AI Detection** - 85%+ accuracy with Random Forest models
- ⚡ **Real-Time Analysis** - Instant URL and email scanning
- 🔐 **Authentication** - Secure JWT-based user system with bcrypt
- 📊 **Dashboard** - Track scans, view history, and analytics
- 🎨 **Modern UI** - Dark cyberpunk theme with smooth animations
- 📱 **Responsive** - Works seamlessly on all devices

---

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ and npm
- Python 3.8+ with pip

### One-Click Start (Windows)
```bash
git clone https://github.com/pushkargupta75/PhishGuard.git
cd PhishGuard
start.bat
```

### Manual Start

**Backend:**
```bash
cd backend
pip install -r requirements.txt
python api\main.py
```

**Frontend (new terminal):**
```bash
cd frontend
npm install
npm run dev
```

**Access:** 
- Frontend: `http://localhost:5173`
- Backend API: `http://localhost:8000`
- API Docs: `http://localhost:8000/docs`

---

## 📁 Project Structure

```
PhishGuard/
├── frontend/               # React + Vite frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   ├── context/       # Auth context
│   │   └── utils/         # API utilities
│   └── package.json
├── backend/               # FastAPI backend
│   ├── api/
│   │   └── main.py       # FastAPI application
│   ├── auth/             # JWT authentication
│   ├── database/         # SQLAlchemy models
│   ├── models/           # ML models (.pkl files)
│   └── requirements.txt
├── start.bat             # Windows launcher
└── README.md
```

---

## 🔌 API Endpoints

### Authentication
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/signup` | POST | Register new user |
| `/api/auth/login` | POST | Login user |
| `/api/auth/me` | GET | Get current user |
| `/api/auth/profile` | PUT | Update profile |

### Scanning
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/analyze/url` | POST | Scan URL for phishing |
| `/api/analyze/email` | POST | Scan email content |

### History
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/history` | GET | Get scan history |
| `/api/history/stats` | GET | Get user statistics |
| `/api/history/save` | POST | Save scan to history |
| `/api/history/{id}` | DELETE | Delete scan |

### Example Request
```bash
curl -X POST "http://localhost:8000/api/analyze/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "http://paypal-verify-account.com"}'
```

**Response:**
```json
{
  "classification": "malicious",
  "confidence": 0.92,
  "highlighted_text": "http://**paypal**-**verify**-**account**.com",
  "explanation": "URL shows suspicious keywords, insecure connection commonly found in phishing attempts.",
  "type": "url"
}
```

---

## 📖 Usage

1. **Sign Up** - Create a new account at `http://localhost:5173/signup`
2. **Login** - Access your dashboard
3. **Scan** - Enter a URL or paste email text
4. **Results** - View instant phishing detection results
5. **History** - Track all your scans in the dashboard

### Test Examples

**Phishing URL:** `http://paypal-secure-login-verify-account.com`

**Phishing Email:** 
```
URGENT! Your account has been suspended. 
Click here immediately to verify your password 
or your account will be terminated within 24 hours!
```

---

## 🤖 Machine Learning Models

- **Algorithm:** Random Forest Classifier with 100 estimators
- **URL Model:** 
  - Accuracy: 85%+
  - Features: 15 (URL length, dots, hyphens, suspicious keywords, HTTPS, etc.)
  - Dataset: 450K URLs from [Kaggle](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)
  
- **Email Model:** 
  - Accuracy: 82%+
  - Features: 13 (text length, urgency words, threat language, uppercase ratio, etc.)
  - Dataset: Email phishing from [Kaggle](https://www.kaggle.com/datasets/ethancratchley/email-phishing-dataset)

---

## 🛠️ Tech Stack

**Frontend:**
- React 19.2 with Vite
- Tailwind CSS for styling
- Framer Motion for animations
- Axios for API calls
- React Router for navigation

**Backend:**
- Python 3.11
- FastAPI for REST API
- Scikit-learn for ML models
- SQLAlchemy for database ORM
- JWT (python-jose) for authentication
- Bcrypt for password hashing
- Uvicorn as ASGI server

**Database:**
- SQLite (development)
- PostgreSQL ready (production)

---

## 🔧 Configuration

### Backend Configuration
Edit `backend/auth/auth_utils.py` for JWT settings:
```python
SECRET_KEY = "your-secret-key-here"  # CHANGE IN PRODUCTION!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days
```

### Frontend Configuration
Edit `frontend/src/utils/api.js` for API URL:
```javascript
const API_BASE_URL = 'http://localhost:8000';
```

---

## 🐛 Troubleshooting

**Port already in use:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

**Module import errors:**
```bash
cd backend
pip install -r requirements.txt --upgrade
```

**Frontend build errors:**
```bash
cd frontend
rm -rf node_modules
npm install
```

**Database errors:**
Delete `backend/database/phishguard.db` and restart backend to recreate

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Pushkar Gupta**
- GitHub: [@pushkargupta75](https://github.com/pushkargupta75)
- Project Link: [PhishGuard](https://github.com/pushkargupta75/PhishGuard)

---

## 🙏 Acknowledgments

- Kaggle for providing the phishing datasets
- FastAPI for the excellent web framework
- React community for amazing tools and libraries
- All contributors who help improve this project

---

<div align="center">

**⭐ Star this repo if it helped you stay safe online! ⭐**

Built with ❤️ for cybersecurity awareness

</div>
