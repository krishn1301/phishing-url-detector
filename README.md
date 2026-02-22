# 🔒 Phishing URL Detection Using Machine Learning

A complete, end-to-end Python web application that analyses URLs to detect phishing attempts using a trained Random Forest classifier. Built for cybersecurity academic research, this project features robust URL feature extraction, host-based analysis (WHOIS/DNS), and a Safe Preview module for safe analysis of malicious links.

---

## 🚀 Features

- **Machine Learning Classification**: Uses a Random Forest model trained on a balanced synthetic dataset.
- **Advanced Feature Extraction**: Extracts 14 distinct features from any given URL:
  - **URL Features**: Length, Dot/At/Dash/Underscore Counts, IP presence, HTTPS check, Suspicious Keywords count, Subdomain count, and Shannon Entropy.
  - **Host-Based Features**: Domain Age (WHOIS), WHOIS Database Availability, DNS A Record count, and DNS MX Record presence.
- **Safe Preview Module**: For URLs flagged with a risk score > 70%, the app safely fetches the raw HTML (without executing JavaScript or downloading payloads) and scans for:
  - Hidden forms & password fields.
  - External form-action submissions.
  - Suspicious keywords hidden in the page body.
- **Clean UI / Dashboard**: A minimal, responsive white-theme dashboard providing real-time risk scores and visual feature breakdowns.

---

## 📋 Architecture & Data Flow

1. **User Input** → User submits a URL via the web front-end.
2. **Feature Extraction (`utils.py`)** → The backend parses the URL and performs live WHOIS and DNS lookups to generate a 14-element numeric feature vector.
3. **ML Prediction (`app.py`)** → The Flask backend feeds the vector into the pre-trained `phishing_model.pkl`.
4. **Risk Scoring** → The model returns a classification (Legitimate/Phishing) and an exact risk percentage.
5. **Safe Preview** → If Risk > 70%, `BeautifulSoup` scrapes the target's raw HTML for further forensic details.
6. **Result Display (`index.html`)** → The payload is rendered gracefully on the UI.

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

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Packages include: `flask`, `scikit-learn`, `joblib`, `beautifulsoup4`, `requests`, `numpy`, `pandas`, `python-whois`, `dnspython`)*

### 3. Train the Model
The repository does not include the 2000-sample dataset or the compiled `.pkl` model to keep it lightweight. You must generate them first:
```bash
python train_model.py
```
*Expected Output: Data generation logs, followed by model training metrics (Accuracy, Precision, Recall).*

### 4. Run the Web App
```bash
python app.py
```
The server will start at `http://127.0.0.1:5000`.

---

## 📊 Screenshots & Application View

### 🟢 Testing a Legitimate URL
When tested with `https://www.google.com`, the system correctly identifies the domain's long-standing WHOIS records and active DNS, returning a low risk score.

![Legitimate URL Example](https://raw.githubusercontent.com/krishn1301/phishing-url-detector/master/screenshots/legitimate.png)

### 🔴 Testing a Phishing URL
When tested with a suspicious IP-based URL (`http://192.168.1.1/login/verify/account`), the system flags the missing WHOIS/DNS data and the presence of suspicious keywords, returning a high risk score and triggering the **Safe Preview Analysis**.

![Phishing URL Example](https://raw.githubusercontent.com/krishn1301/phishing-url-detector/master/screenshots/phishing.png)

*(Note: Add the screenshots to a `/screenshots` folder in your repo to render them here)*

---

## 🗂️ Project Structure

```text
phishing-url-detector/
│
├── app.py                  # Flask backend & API routing
├── train_model.py          # Dataset generator & Random Forest training script
├── utils.py                # Feature extraction & Safe Preview logic
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files (models, datasets, cache)
│
├── templates/
│   └── index.html          # Web Interface UI
└── static/
    └── styles.css          # Minimalist styling dashboard
```

---

## 🛡️ Security Disclaimer
The **Safe Preview Module** is designed to fetch raw HTML without executing potentially malicious JavaScript payloads. However, excessive testing of active malware/phishing domains should always be done inside a sandboxed environment (e.g., a Virtual Machine) to prevent IP tracking or accidental execution.

---

*Built as an Academic Cybersecurity Mini-Project © 2026*
