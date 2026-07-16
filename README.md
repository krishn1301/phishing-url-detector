# 🛡️ PhishGuard — Phishing URL Detection Using Machine Learning

A professional-grade phishing URL detection system powered by **XGBoost** machine learning. Built for cybersecurity academic research, PhishGuard extracts **22 features** from any URL — including SSL/TLS certificate analysis, redirect chain detection, URL shortener identification, and TLD risk scoring — and classifies it in real-time through a premium dark-themed cybersecurity dashboard.

---

## 🚀 Features

- **XGBoost ML Classifier**: Trained on 10,000 realistic synthetic URLs with 5-fold cross-validation and comprehensive evaluation reports.
- **22-Feature Extraction Pipeline**:
  - **URL-Lexical** (10): Length, dots, dashes, underscores, `@` symbols, IP-address usage, HTTPS, suspicious keywords, subdomain count, Shannon entropy.
  - **Host-Based** (4): Domain age (WHOIS), WHOIS availability, DNS A record count, DNS MX record presence.
  - **Security & Network** (8): URL shortener detection, redirect chain count, SSL certificate validity, SSL days remaining, free/DV certificate detection, TLD risk score, path depth, punycode/IDN detection.
- **Safe Preview Module**: For high-risk URLs (>70%), safely fetches raw HTML and scans for hidden forms, password fields, external form submissions, iframes, and suspicious keywords.
- **Scan History**: SQLite-backed persistence of all scan results with aggregate statistics.
- **Premium Dark UI**: Cybersecurity-inspired dashboard with animated threat gauge, tabbed feature breakdown, scan history sidebar, and micro-animations.

---

## 📋 Architecture & Data Flow

```
User Input → Feature Extraction (22 features) → XGBoost Prediction → Risk Scoring
                                                                         ↓
                                              Safe Preview (if risk > 70%) → SQLite Storage
                                                                         ↓
                                                            Dashboard Rendering
```

1. **User Input** → User submits a URL via the web dashboard.
2. **Feature Extraction (`utils.py`)** → Extracts a 22-element numeric feature vector with live SSL, WHOIS, DNS, and redirect lookups.
3. **ML Prediction (`app.py`)** → XGBoost classifier returns classification + probability.
4. **Threat Classification** → 4-tier system: Safe (≤20%) · Low Risk (≤50%) · Medium Risk (≤75%) · Critical (>75%).
5. **Safe Preview** → If risk > 70%, `BeautifulSoup` scrapes the target's raw HTML for forensic details.
6. **Persistence** → Every scan saved to SQLite with full feature data.
7. **Dashboard** → Results rendered with animated gauge, colored badges, and feature tabs.

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/krishn1301/phishing-url-detector.git
cd phishing-url-detector
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Packages:** `flask`, `scikit-learn`, `joblib`, `beautifulsoup4`, `requests`, `numpy`, `pandas`, `python-whois`, `dnspython`, `xgboost`, `tldextract`, `seaborn`, `matplotlib`

### 4. Train the Model
```bash
python train_model.py
```
This will:
- Generate 10,000 realistic URLs (balanced legitimate/phishing)
- Extract 22 features per URL
- Train an XGBoost classifier with 5-fold cross-validation
- Save evaluation plots to `reports/` (confusion matrix, ROC curve, feature importance)
- Save the model to `phishing_model.pkl`

### 5. Run the Web App
```bash
python app.py
```
Open `http://127.0.0.1:5000` in your browser.

---

## 📊 Screenshots

### Dashboard — Initial State
![Dashboard](screenshots/dashboard.png)

### 🟢 Legitimate URL Scan
`https://www.google.com` → **Safe** (0.0% risk)

![Legitimate URL](screenshots/legitimate.png)

### 🔴 Phishing URL Scan
`http://192.168.1.1/login/verify/account` → **Critical** (100% risk)

![Phishing URL](screenshots/phishing.png)

---

## 📈 Model Evaluation

The XGBoost model achieves excellent performance on the test set:

| Metric | Score |
|---|---|
| **Accuracy** | 1.0000 |
| **Precision** | 1.0000 |
| **Recall** | 1.0000 |
| **F1-Score** | 1.0000 |
| **ROC AUC** | 1.0000 |

Evaluation plots are saved to `reports/`:
- `confusion_matrix.png`
- `roc_curve.png`
- `feature_importance.png`

### Top Features by Importance
1. SSL Days Remaining
2. Domain Age (days)
3. Suspicious Keywords
4. DNS A Records

---

## 🗂️ Project Structure

```text
phishing-url-detector/
│
├── app.py                  # Flask backend, SQLite history, API routes
├── train_model.py          # URL generation, XGBoost training, evaluation plots
├── utils.py                # 22-feature extraction & Safe Preview module
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files
│
├── templates/
│   └── index.html          # Cybersecurity dashboard UI
├── static/
│   └── styles.css          # Dark theme with glassmorphism & animations
├── reports/                # Auto-generated evaluation plots
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
└── screenshots/            # App screenshots for README
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the dashboard |
| `POST` | `/predict` | Scan a URL (body: `{"url": "..."}`) |
| `GET` | `/api/history` | Last 50 scan results |
| `DELETE` | `/api/history` | Clear all scan history |
| `GET` | `/api/stats` | Aggregate scan statistics |

---

## 🛡️ Security Disclaimer

The **Safe Preview Module** fetches raw HTML without executing JavaScript. However, testing active phishing domains should be done inside a sandboxed environment (VM/container) to prevent IP tracking or accidental execution of malicious payloads.

---

*Built as an Academic Cybersecurity Mini-Project © 2026*
