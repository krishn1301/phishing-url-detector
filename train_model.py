"""
train_model.py - Model Training Script
========================================
Generates a synthetic phishing/legitimate URL dataset, extracts features
using utils.extract_features(), trains a Random Forest classifier, prints
evaluation metrics, and saves the model as phishing_model.pkl.

Usage:
    python train_model.py
"""

import os
import random
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

from utils import extract_features

# Suppress convergence / future warnings during training
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Synthetic Dataset Generation
# ---------------------------------------------------------------------------

# Common legitimate domains
LEGIT_DOMAINS = [
    "google.com", "facebook.com", "amazon.com", "microsoft.com",
    "apple.com", "github.com", "stackoverflow.com", "wikipedia.org",
    "youtube.com", "twitter.com", "linkedin.com", "reddit.com",
    "netflix.com", "spotify.com", "dropbox.com", "medium.com",
    "bbc.co.uk", "cnn.com", "nytimes.com", "washingtonpost.com",
    "instagram.com", "pinterest.com", "tumblr.com", "quora.com",
    "yahoo.com", "bing.com", "duckduckgo.com", "twitch.tv",
    "slack.com", "zoom.us", "adobe.com", "salesforce.com",
]

LEGIT_PATHS = [
    "/", "/about", "/contact", "/products", "/services",
    "/help", "/faq", "/blog", "/news", "/careers",
    "/docs", "/support", "/pricing", "/features", "/terms",
    "/privacy", "/settings", "/profile", "/dashboard", "/search",
]

# Phishing-style components
PHISH_DOMAINS = [
    "192.168.1.1", "10.0.0.1", "172.16.0.1",
    "secure-login-verify.com", "accounts-update-info.net",
    "bank-secure-login.com", "paypal-verify-account.com",
    "login-update-secure.net", "verify-account-bank.com",
    "secure-signin-confirm.org", "update-password-bank.net",
    "account-verify-login.com", "signin-secure-confirm.net",
]

PHISH_SUBDOMAINS = [
    "login.", "secure.", "verify.", "account.", "update.",
    "signin.", "confirm.", "bank.", "password.", "auth.",
    "mail.", "web.", "portal.", "service.", "client.",
]

PHISH_PATHS = [
    "/login", "/verify", "/secure", "/update", "/confirm",
    "/signin", "/account/login", "/bank/verify", "/password/reset",
    "/login/verify/account", "/secure/update/password",
    "/confirm/identity", "/auth/signin", "/verify-email",
    "/account-update", "/secure-login", "/bank-verify",
]

PHISH_PARAMS = [
    "?user=admin&token=abc123", "?login=true&verify=1",
    "?secure=yes&bank=update", "?id=12345&confirm=true",
    "?email=user@bank.com", "?redirect=http://evil.com",
    "?session=xyz&action=verify", "",
]


def _generate_legit_url() -> str:
    """Generate a realistic legitimate URL."""
    domain = random.choice(LEGIT_DOMAINS)
    path = random.choice(LEGIT_PATHS)
    scheme = random.choice(["https://", "https://", "https://", "http://"])  # Mostly HTTPS
    subdomain = random.choice(["www.", "www.", "www.", ""])  # Mostly www
    return f"{scheme}{subdomain}{domain}{path}"


def _generate_phish_url() -> str:
    """Generate a realistic phishing-style URL."""
    domain = random.choice(PHISH_DOMAINS)
    path = random.choice(PHISH_PATHS)
    params = random.choice(PHISH_PARAMS)
    scheme = random.choice(["http://", "http://", "http://", "https://"])  # Mostly HTTP
    # Add 1-3 subdomains
    num_subdomains = random.randint(1, 3)
    subdomains = "".join(random.choices(PHISH_SUBDOMAINS, k=num_subdomains))
    return f"{scheme}{subdomains}{domain}{path}{params}"


def generate_dataset(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generate a balanced synthetic dataset of legitimate and phishing URLs.

    Args:
        n_samples: Total number of samples (split 50/50).

    Returns:
        DataFrame with columns ['url', 'label'] where label 1 = phishing.
    """
    half = n_samples // 2
    urls, labels = [], []

    for _ in range(half):
        urls.append(_generate_legit_url())
        labels.append(0)

    for _ in range(half):
        urls.append(_generate_phish_url())
        labels.append(1)

    df = pd.DataFrame({"url": urls, "label": labels})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------

def main():
    dataset_path = os.path.join(os.path.dirname(__file__) or ".", "phishing_dataset.csv")
    model_path = os.path.join(os.path.dirname(__file__) or ".", "phishing_model.pkl")

    # Step 1 — Dataset (always regenerate to pick up feature changes)
    if os.path.exists(dataset_path):
        os.remove(dataset_path)
    print("[*] Generating synthetic dataset (2 000 samples) ...")
    df = generate_dataset(2000)
    df.to_csv(dataset_path, index=False)
    print(f"[+] Dataset saved to {dataset_path}")

    print(f"[*] Dataset shape: {df.shape}")
    print(f"    Legitimate: {(df['label'] == 0).sum()}  |  Phishing: {(df['label'] == 1).sum()}")

    # Step 2 — Feature extraction (offline mode, no network calls)
    print("[*] Extracting features ...")
    X = np.array([extract_features(url, live_lookup=False) for url in df["url"]])
    y = df["label"].values

    # Step 2b — Simulate host-based features for synthetic data
    #   Indices: 10=domain_age_days, 11=whois_available,
    #            12=dns_a_record_count, 13=dns_has_mx
    print("[*] Simulating host-based features for synthetic data ...")
    for i in range(len(X)):
        if y[i] == 0:  # Legitimate
            X[i][10] = float(random.randint(365, 9000))    # Old domain (1-25 years)
            X[i][11] = 1.0                                 # WHOIS available
            X[i][12] = float(random.randint(1, 6))         # Multiple A records
            X[i][13] = 1.0                                 # MX record exists
        else:          # Phishing
            X[i][10] = float(random.choice([-1, *range(0, 90)]))  # New or unknown
            X[i][11] = float(random.choice([0, 0, 0, 1]))         # Usually no WHOIS
            X[i][12] = float(random.choice([0, 0, 1]))             # Few/no A records
            X[i][13] = float(random.choice([0, 0, 0, 1]))         # Usually no MX

    # Step 3 — Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    print(f"[*] Train size: {len(X_train)}  |  Test size: {len(X_test)}")

    # Step 4 — Train Random Forest
    print("[*] Training Random Forest classifier ...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Step 5 — Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    print("\n===== Evaluation Metrics =====")
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print("==============================\n")

    # Step 6 — Save model
    joblib.dump(model, model_path)
    print(f"[+] Model saved to {model_path}")


if __name__ == "__main__":
    main()
