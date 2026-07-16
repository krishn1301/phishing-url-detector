"""
app.py - Flask Backend for Phishing URL Detection
===================================================
Loads the trained XGBoost model, serves a cybersecurity dashboard,
and persists scan history in SQLite.

Usage:
    python app.py
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone

import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify, g

from utils import (
    extract_features, safe_preview,
    FEATURE_NAMES, URL_FEATURES, HOST_FEATURES, SECURITY_FEATURES,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "phishing_model.pkl")
DB_PATH = os.path.join(BASE_DIR, "scan_history.db")
RISK_THRESHOLD = 0.70  # 70 % — trigger safe preview above this

# Threat-level tiers
THREAT_LEVELS = [
    (20, "Safe", "safe"),
    (50, "Low Risk", "low"),
    (75, "Medium Risk", "medium"),
    (100, "Critical", "critical"),
]

# ---------------------------------------------------------------------------
# App Setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Load the trained model at startup
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    app.logger.info("Model loaded from %s", MODEL_PATH)
else:
    model = None
    app.logger.warning(
        "Model file not found at %s. Run train_model.py first.", MODEL_PATH
    )


# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _init_db():
    """Create the scan_history table if it doesn't exist."""
    db = sqlite3.connect(DB_PATH)
    db.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            url             TEXT    NOT NULL,
            classification  TEXT    NOT NULL,
            threat_level    TEXT    NOT NULL,
            risk_score      REAL    NOT NULL,
            features_json   TEXT,
            preview_json    TEXT,
            scanned_at      TEXT    NOT NULL
        )
    """)
    db.commit()
    db.close()


_init_db()


def _get_db():
    """Return a per-request SQLite connection."""
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def _close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def _save_scan(url, classification, threat_level, risk, features, preview):
    """Persist a scan result."""
    db = _get_db()
    db.execute(
        """INSERT INTO scans
           (url, classification, threat_level, risk_score,
            features_json, preview_json, scanned_at)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            url,
            classification,
            threat_level,
            risk,
            json.dumps(features),
            json.dumps(preview) if preview else None,
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    db.commit()


def _classify_threat(risk_pct: float) -> tuple:
    """Return (label, css_class) for a given risk percentage."""
    for threshold, label, css in THREAT_LEVELS:
        if risk_pct <= threshold:
            return label, css
    return "Critical", "critical"


def _evaluate_red_flags(feature_dict: dict) -> dict:
    """
    Evaluate each feature and return a dict of red flags.
    Each entry: feature_name -> { "flagged": bool, "severity": str, "reason": str }
    severity: "danger" | "warning" | "safe"
    """
    flags = {}

    # --- URL Length ---
    val = feature_dict.get("URL Length", 0)
    if val > 75:
        flags["URL Length"] = {"flagged": True, "severity": "danger",
            "reason": f"URL is {int(val)} characters long. Phishing URLs are often excessively long to hide the real domain in the address bar."}
    elif val > 54:
        flags["URL Length"] = {"flagged": True, "severity": "warning",
            "reason": f"URL is {int(val)} characters — moderately long. Attackers sometimes pad URLs to obscure malicious domains."}
    else:
        flags["URL Length"] = {"flagged": False, "severity": "safe",
            "reason": "URL length is within normal range."}

    # --- Dot Count ---
    val = feature_dict.get("Dot Count", 0)
    if val > 4:
        flags["Dot Count"] = {"flagged": True, "severity": "danger",
            "reason": f"URL contains {int(val)} dots. Excessive dots often indicate subdomain abuse to mimic legitimate domains."}
    elif val > 3:
        flags["Dot Count"] = {"flagged": True, "severity": "warning",
            "reason": f"URL has {int(val)} dots — more than typical. Could indicate subdomain spoofing."}
    else:
        flags["Dot Count"] = {"flagged": False, "severity": "safe",
            "reason": "Dot count is normal for a standard URL."}

    # --- @ Count ---
    val = feature_dict.get("@ Count", 0)
    if val > 0:
        flags["@ Count"] = {"flagged": True, "severity": "danger",
            "reason": "URL contains an '@' symbol. This is a classic phishing trick — everything before @ is ignored by the browser, hiding the real destination."}
    else:
        flags["@ Count"] = {"flagged": False, "severity": "safe",
            "reason": "No '@' symbol found. This is expected for legitimate URLs."}

    # --- Dash Count ---
    val = feature_dict.get("Dash Count", 0)
    if val > 3:
        flags["Dash Count"] = {"flagged": True, "severity": "danger",
            "reason": f"URL has {int(val)} dashes. Phishing domains frequently use dashes to mimic legitimate domains (e.g. secure-login-bank.com)."}
    elif val > 2:
        flags["Dash Count"] = {"flagged": True, "severity": "warning",
            "reason": f"URL has {int(val)} dashes — slightly above average. Could be used for domain impersonation."}
    else:
        flags["Dash Count"] = {"flagged": False, "severity": "safe",
            "reason": "Dash count is within normal range."}

    # --- Underscore Count ---
    val = feature_dict.get("Underscore Count", 0)
    if val > 2:
        flags["Underscore Count"] = {"flagged": True, "severity": "warning",
            "reason": f"URL contains {int(val)} underscores. Legitimate domains rarely use underscores; this is more common in phishing URLs."}
    else:
        flags["Underscore Count"] = {"flagged": False, "severity": "safe",
            "reason": "Underscore count is within normal range."}

    # --- Has IP Address ---
    val = feature_dict.get("Has IP Address", 0)
    if val == 1:
        flags["Has IP Address"] = {"flagged": True, "severity": "danger",
            "reason": "URL uses a raw IP address instead of a domain name. Legitimate websites almost always use domain names — IP-based URLs are a major phishing indicator."}
    else:
        flags["Has IP Address"] = {"flagged": False, "severity": "safe",
            "reason": "URL uses a proper domain name, not a raw IP address."}

    # --- Has HTTPS ---
    val = feature_dict.get("Has HTTPS", 0)
    if val == 0:
        flags["Has HTTPS"] = {"flagged": True, "severity": "danger",
            "reason": "URL does not use HTTPS. Without encryption, any data you enter can be intercepted. Legitimate sites use HTTPS for security."}
    else:
        flags["Has HTTPS"] = {"flagged": False, "severity": "safe",
            "reason": "URL uses HTTPS encryption — data in transit is protected."}

    # --- Suspicious Keywords ---
    val = feature_dict.get("Suspicious Keywords", 0)
    if val > 2:
        flags["Suspicious Keywords"] = {"flagged": True, "severity": "danger",
            "reason": f"URL contains {int(val)} suspicious keywords (e.g. login, verify, account, bank). Phishers embed these terms to appear legitimate."}
    elif val > 0:
        flags["Suspicious Keywords"] = {"flagged": True, "severity": "warning",
            "reason": f"URL contains {int(val)} suspicious keyword(s). While sometimes legitimate, these terms are commonly used in phishing lures."}
    else:
        flags["Suspicious Keywords"] = {"flagged": False, "severity": "safe",
            "reason": "No suspicious keywords detected in the URL."}

    # --- Subdomain Count ---
    val = feature_dict.get("Subdomain Count", 0)
    if val > 2:
        flags["Subdomain Count"] = {"flagged": True, "severity": "danger",
            "reason": f"URL has {int(val)} subdomains. Deep subdomain nesting is a common tactic to hide the real domain (e.g. login.secure.bank.evil.com)."}
    elif val > 1:
        flags["Subdomain Count"] = {"flagged": True, "severity": "warning",
            "reason": f"URL has {int(val)} subdomains — slightly unusual. May indicate subdomain abuse."}
    else:
        flags["Subdomain Count"] = {"flagged": False, "severity": "safe",
            "reason": "Subdomain count is normal."}

    # --- Entropy ---
    val = feature_dict.get("Entropy", 0)
    if val > 4.5:
        flags["Entropy"] = {"flagged": True, "severity": "danger",
            "reason": f"URL entropy is {val:.2f} (high randomness). Phishing URLs often contain random character sequences generated by automated tools."}
    elif val > 4.0:
        flags["Entropy"] = {"flagged": True, "severity": "warning",
            "reason": f"URL entropy is {val:.2f} — moderately high. This could indicate randomly generated URL components."}
    else:
        flags["Entropy"] = {"flagged": False, "severity": "safe",
            "reason": f"URL entropy ({val:.2f}) is within normal range — the URL appears human-readable."}

    # --- Domain Age ---
    val = feature_dict.get("Domain Age (days)", -1)
    if val < 0:
        flags["Domain Age (days)"] = {"flagged": True, "severity": "warning",
            "reason": "Domain age could not be determined. This may indicate a very new or suspicious domain."}
    elif val < 30:
        flags["Domain Age (days)"] = {"flagged": True, "severity": "danger",
            "reason": f"Domain is only {int(val)} days old. Most phishing sites are set up on newly registered domains that are days or weeks old."}
    elif val < 180:
        flags["Domain Age (days)"] = {"flagged": True, "severity": "warning",
            "reason": f"Domain is {int(val)} days old — relatively new. Established brands typically have domains registered for years."}
    else:
        flags["Domain Age (days)"] = {"flagged": False, "severity": "safe",
            "reason": f"Domain has been registered for {int(val)} days — well-established."}

    # --- WHOIS Available ---
    val = feature_dict.get("WHOIS Available", 0)
    if val == 0:
        flags["WHOIS Available"] = {"flagged": True, "severity": "warning",
            "reason": "WHOIS data is not available for this domain. Phishing sites often use privacy protection or registrars that hide WHOIS info."}
    else:
        flags["WHOIS Available"] = {"flagged": False, "severity": "safe",
            "reason": "WHOIS data is publicly available — domain registration is transparent."}

    # --- DNS A Records ---
    val = feature_dict.get("DNS A Records", 0)
    if val == 0:
        flags["DNS A Records"] = {"flagged": True, "severity": "danger",
            "reason": "No DNS A records found. The domain may not be properly configured or could be a dead/malicious domain."}
    else:
        flags["DNS A Records"] = {"flagged": False, "severity": "safe",
            "reason": f"Domain has {int(val)} DNS A record(s) — properly configured."}

    # --- DNS Has MX ---
    val = feature_dict.get("DNS Has MX", 0)
    if val == 0:
        flags["DNS Has MX"] = {"flagged": True, "severity": "warning",
            "reason": "Domain has no MX records (no email capability). Legitimate businesses typically have email infrastructure set up."}
    else:
        flags["DNS Has MX"] = {"flagged": False, "severity": "safe",
            "reason": "Domain has MX records — email infrastructure exists, typical of legitimate organizations."}

    # --- Is Shortened URL ---
    val = feature_dict.get("Is Shortened URL", 0)
    if val == 1:
        flags["Is Shortened URL"] = {"flagged": True, "severity": "danger",
            "reason": "URL is from a known URL shortener service. Shortened URLs hide the true destination and are frequently used in phishing campaigns."}
    else:
        flags["Is Shortened URL"] = {"flagged": False, "severity": "safe",
            "reason": "URL is not shortened — the full destination is visible."}

    # --- Redirect Count ---
    val = feature_dict.get("Redirect Count", 0)
    if val < 0:
        flags["Redirect Count"] = {"flagged": True, "severity": "warning",
            "reason": "Redirect chain could not be followed. The server may be unreachable or blocking requests."}
    elif val > 3:
        flags["Redirect Count"] = {"flagged": True, "severity": "danger",
            "reason": f"URL has {int(val)} redirects. Excessive redirects are used to evade security filters and obscure the final malicious destination."}
    elif val > 1:
        flags["Redirect Count"] = {"flagged": True, "severity": "warning",
            "reason": f"URL has {int(val)} redirects — slightly more than usual. Multiple redirects can be used to hide the true destination."}
    else:
        flags["Redirect Count"] = {"flagged": False, "severity": "safe",
            "reason": "Minimal or no redirects — the URL goes directly to its destination."}

    # --- SSL Valid ---
    val = feature_dict.get("SSL Valid", 0)
    if val == 0:
        flags["SSL Valid"] = {"flagged": True, "severity": "danger",
            "reason": "SSL certificate is invalid or expired. This means the connection is not secure and the site's identity cannot be verified."}
    else:
        flags["SSL Valid"] = {"flagged": False, "severity": "safe",
            "reason": "SSL certificate is valid — the site's identity has been verified by a Certificate Authority."}

    # --- SSL Days Remaining ---
    val = feature_dict.get("SSL Days Remaining", -1)
    if val < 0:
        flags["SSL Days Remaining"] = {"flagged": True, "severity": "danger",
            "reason": "SSL certificate information is unavailable or the certificate has expired."}
    elif val < 30:
        flags["SSL Days Remaining"] = {"flagged": True, "severity": "warning",
            "reason": f"SSL certificate expires in {int(val)} days. Phishing sites often use short-lived certificates."}
    else:
        flags["SSL Days Remaining"] = {"flagged": False, "severity": "safe",
            "reason": f"SSL certificate is valid for {int(val)} more days."}

    # --- Free Certificate ---
    val = feature_dict.get("Free Certificate", 0)
    if val == 1:
        flags["Free Certificate"] = {"flagged": True, "severity": "warning",
            "reason": "Site uses a free SSL certificate (e.g. Let's Encrypt). While common for legitimate sites too, phishing sites overwhelmingly use free certificates due to zero cost and instant issuance."}
    else:
        flags["Free Certificate"] = {"flagged": False, "severity": "safe",
            "reason": "Site uses a paid/organization-validated SSL certificate."}

    # --- TLD Risk Score ---
    val = feature_dict.get("TLD Risk Score", 0)
    if val >= 0.7:
        flags["TLD Risk Score"] = {"flagged": True, "severity": "danger",
            "reason": f"TLD risk score is {val:.1f} (high). This top-level domain (.tk, .ml, .xyz, etc.) is heavily abused in phishing campaigns due to free or cheap registration."}
    elif val >= 0.4:
        flags["TLD Risk Score"] = {"flagged": True, "severity": "warning",
            "reason": f"TLD risk score is {val:.1f} (moderate). This top-level domain has some history of abuse in phishing."}
    else:
        flags["TLD Risk Score"] = {"flagged": False, "severity": "safe",
            "reason": "TLD risk score is low — this top-level domain is not commonly associated with phishing."}

    # --- Path Depth ---
    val = feature_dict.get("Path Depth", 0)
    if val > 5:
        flags["Path Depth"] = {"flagged": True, "severity": "warning",
            "reason": f"URL path has {int(val)} levels deep. Deeply nested paths can be used to hide the real content or create deceptive directory structures."}
    else:
        flags["Path Depth"] = {"flagged": False, "severity": "safe",
            "reason": "URL path depth is within normal range."}

    # --- Has Punycode ---
    val = feature_dict.get("Has Punycode", 0)
    if val == 1:
        flags["Has Punycode"] = {"flagged": True, "severity": "danger",
            "reason": "Domain uses Punycode (internationalized domain name). This is a known homograph attack vector — characters from other scripts can look identical to Latin letters (e.g. 'аpple.com' using Cyrillic 'а')."}
    else:
        flags["Has Punycode"] = {"flagged": False, "severity": "safe",
            "reason": "Domain does not use Punycode — no homograph attack risk."}

    return flags


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main dashboard."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accept a URL, extract features, classify with the model, and
    optionally run the safe preview module.
    """
    data = request.get_json(silent=True)
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' field in request body."}), 400

    url = data["url"].strip()
    if not url:
        return jsonify({"error": "URL cannot be empty."}), 400

    if model is None:
        return jsonify({
            "error": "Model not loaded. Please run train_model.py first."
        }), 503

    try:
        # Step 1 — Feature extraction
        features = extract_features(url)
        feature_array = np.array(features).reshape(1, -1)

        # Step 2 — Prediction
        prediction = model.predict(feature_array)[0]
        probabilities = model.predict_proba(feature_array)[0]

        risk = float(probabilities[1]) * 100
        classification = "Phishing" if prediction == 1 else "Legitimate"
        threat_label, threat_css = _classify_threat(risk)

        # Build categorised feature dict
        feature_dict = {
            name: val for name, val in zip(FEATURE_NAMES, features)
        }

        # Step 3 — Red flag analysis
        red_flags = _evaluate_red_flags(feature_dict)

        # Step 4 — Safe preview (only for high-risk URLs)
        preview = None
        if risk >= RISK_THRESHOLD * 100:
            preview = safe_preview(url)

        # Persist to history
        _save_scan(url, classification, threat_label, risk, feature_dict, preview)

        return jsonify({
            "classification": classification,
            "threat_level": threat_label,
            "threat_css": threat_css,
            "risk_percentage": round(risk, 2),
            "features": feature_dict,
            "red_flags": red_flags,
            "feature_groups": {
                "url": URL_FEATURES,
                "host": HOST_FEATURES,
                "security": SECURITY_FEATURES,
            },
            "preview": preview,
        })

    except Exception as exc:
        app.logger.exception("Prediction failed for URL: %s", url)
        return jsonify({"error": f"Prediction failed: {str(exc)}"}), 500


@app.route("/api/history")
def history():
    """Return the last 50 scans."""
    db = _get_db()
    rows = db.execute(
        "SELECT * FROM scans ORDER BY id DESC LIMIT 50"
    ).fetchall()

    scans = []
    for row in rows:
        scans.append({
            "id": row["id"],
            "url": row["url"],
            "classification": row["classification"],
            "threat_level": row["threat_level"],
            "risk_score": row["risk_score"],
            "scanned_at": row["scanned_at"],
        })
    return jsonify(scans)


@app.route("/api/stats")
def stats():
    """Return aggregate scan stats."""
    db = _get_db()
    total = db.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    phishing = db.execute(
        "SELECT COUNT(*) FROM scans WHERE classification = 'Phishing'"
    ).fetchone()[0]
    avg_risk = db.execute(
        "SELECT COALESCE(AVG(risk_score), 0) FROM scans"
    ).fetchone()[0]

    return jsonify({
        "total_scans": total,
        "phishing_detected": phishing,
        "legitimate_detected": total - phishing,
        "avg_risk_score": round(avg_risk, 1),
        "phishing_rate": round((phishing / total * 100) if total > 0 else 0, 1),
    })


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    """Clear all scan history."""
    db = _get_db()
    db.execute("DELETE FROM scans")
    db.commit()
    return jsonify({"message": "History cleared."})


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
