"""
train_model.py - XGBoost Model Training Pipeline
==================================================
Generates a large, realistic synthetic phishing/legitimate URL dataset,
extracts 22 features using utils.extract_features(), trains an XGBoost
classifier with 5-fold cross-validation, generates evaluation plots,
and saves the model.

Usage:
    python train_model.py
"""

import os
import random
import warnings

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    classification_report,
)
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from utils import extract_features, FEATURE_NAMES

# Suppress warnings during training
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "phishing_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "phishing_model.pkl")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# ---------------------------------------------------------------------------
# Realistic URL Templates (much larger pool than before)
# ---------------------------------------------------------------------------

# Top 100+ legitimate domains
LEGIT_DOMAINS = [
    "google.com", "facebook.com", "amazon.com", "microsoft.com",
    "apple.com", "github.com", "stackoverflow.com", "wikipedia.org",
    "youtube.com", "twitter.com", "linkedin.com", "reddit.com",
    "netflix.com", "spotify.com", "dropbox.com", "medium.com",
    "bbc.co.uk", "cnn.com", "nytimes.com", "washingtonpost.com",
    "instagram.com", "pinterest.com", "quora.com", "yahoo.com",
    "bing.com", "duckduckgo.com", "twitch.tv", "slack.com",
    "zoom.us", "adobe.com", "salesforce.com", "shopify.com",
    "wordpress.com", "blogger.com", "tumblr.com", "flickr.com",
    "paypal.com", "stripe.com", "squarespace.com", "wix.com",
    "airbnb.com", "booking.com", "uber.com", "lyft.com",
    "tesla.com", "samsung.com", "intel.com", "nvidia.com",
    "oracle.com", "ibm.com", "cisco.com", "vmware.com",
    "atlassian.com", "jira.com", "bitbucket.org", "gitlab.com",
    "heroku.com", "cloudflare.com", "aws.amazon.com",
    "azure.microsoft.com", "cloud.google.com", "digitalocean.com",
    "mongodb.com", "redis.io", "elastic.co", "grafana.com",
    "docker.com", "kubernetes.io", "npmjs.com", "pypi.org",
    "kaggle.com", "coursera.org", "udemy.com", "edx.org",
    "khanacademy.org", "mit.edu", "stanford.edu", "harvard.edu",
    "nytimes.com", "bbc.com", "theguardian.com", "reuters.com",
    "bloomberg.com", "forbes.com", "techcrunch.com", "wired.com",
    "arstechnica.com", "theverge.com", "engadget.com",
    "imdb.com", "rottentomatoes.com", "goodreads.com",
    "etsy.com", "ebay.com", "walmart.com", "target.com",
    "bestbuy.com", "homedepot.com", "lowes.com", "ikea.com",
    "zara.com", "hm.com", "nike.com", "adidas.com",
]

LEGIT_PATHS = [
    "/", "/about", "/contact", "/products", "/services",
    "/help", "/faq", "/blog", "/news", "/careers",
    "/docs", "/support", "/pricing", "/features", "/terms",
    "/privacy", "/settings", "/profile", "/dashboard", "/search",
    "/explore", "/trending", "/popular", "/categories",
    "/learn", "/courses", "/tutorials", "/guide",
    "/api/v1", "/api/v2", "/developers", "/partners",
    "/press", "/media", "/events", "/community",
    "/store", "/shop", "/cart", "/checkout",
]

# Phishing patterns modeled after real-world attack campaigns
PHISH_BASE_DOMAINS = [
    "secure-login-verify.com", "accounts-update-info.net",
    "bank-secure-login.com", "paypal-verify-account.com",
    "login-update-secure.net", "verify-account-bank.com",
    "secure-signin-confirm.org", "update-password-bank.net",
    "account-verify-login.com", "signin-secure-confirm.net",
    "support-alert-verify.com", "security-check-update.net",
    "password-reset-now.com", "urgent-account-action.org",
    "verify-your-identity.net", "confirm-account-info.com",
    "secure-payment-verify.org", "update-billing-info.net",
    "amazon-order-confirm.com", "apple-id-verify.net",
    "microsoft-account-alert.com", "google-security-check.net",
    "netflix-billing-update.com", "facebook-login-verify.org",
]

# Risky TLDs for phishing
PHISH_RISKY_TLDS = [".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top",
                     ".work", ".click", ".loan", ".icu", ".buzz"]

PHISH_SUBDOMAINS = [
    "login.", "secure.", "verify.", "account.", "update.",
    "signin.", "confirm.", "bank.", "password.", "auth.",
    "mail.", "web.", "portal.", "service.", "client.",
    "support.", "billing.", "security.", "alert.", "admin.",
    "help.", "recovery.", "notification.", "manage.", "access.",
]

PHISH_PATHS = [
    "/login", "/verify", "/secure", "/update", "/confirm",
    "/signin", "/account/login", "/bank/verify", "/password/reset",
    "/login/verify/account", "/secure/update/password",
    "/confirm/identity", "/auth/signin", "/verify-email",
    "/account-update", "/secure-login", "/bank-verify",
    "/customer/verify/identity", "/billing/update/payment",
    "/security/alert/confirm", "/suspend/review/appeal",
    "/wallet/verify/transaction", "/payment/confirm/order",
]

PHISH_PARAMS = [
    "?user=admin&token=abc123", "?login=true&verify=1",
    "?secure=yes&bank=update", "?id=12345&confirm=true",
    "?email=user@bank.com", "?redirect=http://evil.com",
    "?session=xyz&action=verify", "",
    "?ref=email&src=alert", "?code=URGENT&act=now",
    "?token=expired&renew=true", "",
    "?uid=874512&verify=account", "",
]

# IP addresses commonly used in phishing
PHISH_IPS = [
    "192.168.1.1", "10.0.0.1", "172.16.0.1", "185.234.219.47",
    "91.219.236.18", "45.33.32.156", "104.248.51.87",
    "138.197.138.255", "167.99.36.123", "206.189.98.54",
]


# ---------------------------------------------------------------------------
# URL Generation
# ---------------------------------------------------------------------------

def _generate_legit_url() -> str:
    """Generate a realistic legitimate URL."""
    domain = random.choice(LEGIT_DOMAINS)
    path = random.choice(LEGIT_PATHS)
    scheme = random.choices(["https://", "http://"], weights=[0.9, 0.1])[0]
    subdomain = random.choices(["www.", ""], weights=[0.7, 0.3])[0]
    return f"{scheme}{subdomain}{domain}{path}"


def _generate_phish_url() -> str:
    """Generate a realistic phishing-style URL."""
    style = random.choice(["domain", "ip", "risky_tld", "typosquat"])

    if style == "ip":
        ip = random.choice(PHISH_IPS)
        path = random.choice(PHISH_PATHS)
        params = random.choice(PHISH_PARAMS)
        return f"http://{ip}{path}{params}"

    elif style == "risky_tld":
        word1 = random.choice(["secure", "login", "verify", "update", "bank",
                                "account", "confirm", "paypal", "alert"])
        word2 = random.choice(["now", "today", "info", "check", "help",
                                "support", "online", "web", "portal"])
        sep = random.choice(["-", ""])
        tld = random.choice(PHISH_RISKY_TLDS)
        num_subs = random.randint(0, 2)
        subs = "".join(random.choices(PHISH_SUBDOMAINS, k=num_subs))
        path = random.choice(PHISH_PATHS)
        params = random.choice(PHISH_PARAMS)
        return f"http://{subs}{word1}{sep}{word2}{tld}{path}{params}"

    elif style == "typosquat":
        # Mimic known brands with typos
        typos = [
            "gooogle.com", "faceb00k.com", "arnazon.com", "mlcrosoft.com",
            "app1e.com", "paypa1.com", "netf1ix.com", "arnazon-login.com",
            "goog1e-verify.com", "microsft-account.com", "arnazon-support.net",
            "appie-id-verify.com", "paypa1-secure.com", "linkedln-login.com",
        ]
        domain = random.choice(typos)
        num_subs = random.randint(0, 2)
        subs = "".join(random.choices(PHISH_SUBDOMAINS, k=num_subs))
        path = random.choice(PHISH_PATHS)
        scheme = random.choices(["http://", "https://"], weights=[0.7, 0.3])[0]
        return f"{scheme}{subs}{domain}{path}"

    else:  # domain style
        domain = random.choice(PHISH_BASE_DOMAINS)
        path = random.choice(PHISH_PATHS)
        params = random.choice(PHISH_PARAMS)
        scheme = random.choices(["http://", "https://"], weights=[0.65, 0.35])[0]
        num_subs = random.randint(1, 3)
        subs = "".join(random.choices(PHISH_SUBDOMAINS, k=num_subs))
        return f"{scheme}{subs}{domain}{path}{params}"


def generate_dataset(n_samples: int = 10000) -> pd.DataFrame:
    """Generate a balanced synthetic dataset."""
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
# Evaluation Plot Helpers
# ---------------------------------------------------------------------------

def _plot_confusion_matrix(y_true, y_pred, save_path):
    """Generate and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5.5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"],
        annot_kws={"size": 16},
        linewidths=0.5,
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("Actual", fontsize=12)
    plt.xlabel("Predicted", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ✓ Saved {save_path}")


def _plot_roc_curve(y_true, y_proba, save_path):
    """Generate and save a ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5.5))
    plt.plot(fpr, tpr, color="#4f46e5", lw=2.5,
             label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="#9ca3af", lw=1.5, linestyle="--",
             label="Random Classifier")
    plt.fill_between(fpr, tpr, alpha=0.1, color="#4f46e5")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ✓ Saved {save_path}")


def _plot_feature_importance(model, feature_names, save_path):
    """Generate and save a feature importance bar chart."""
    importances = model.feature_importances_
    indices = np.argsort(importances)

    plt.figure(figsize=(9, 7))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(indices)))
    plt.barh(range(len(indices)), importances[indices], color=colors)
    plt.yticks(range(len(indices)),
               [feature_names[i] for i in indices], fontsize=10)
    plt.xlabel("Importance Score", fontsize=12)
    plt.title("Feature Importance (XGBoost)", fontsize=14, fontweight="bold")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"    ✓ Saved {save_path}")


# ---------------------------------------------------------------------------
# Main Training Pipeline
# ---------------------------------------------------------------------------

def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # ── Step 1: Generate dataset ──────────────────────────────────────
    if os.path.exists(DATASET_PATH):
        os.remove(DATASET_PATH)
    print("=" * 60)
    print("  PHISHING URL DETECTOR — MODEL TRAINING PIPELINE")
    print("=" * 60)
    print()
    print("[1/6] Generating realistic synthetic dataset (10,000 samples) ...")
    df = generate_dataset(10000)
    df.to_csv(DATASET_PATH, index=False)
    print(f"      Dataset saved: {DATASET_PATH}")
    print(f"      Shape: {df.shape}")
    print(f"      Legitimate: {(df['label'] == 0).sum()}  |  Phishing: {(df['label'] == 1).sum()}")
    print()

    # ── Step 2: Feature extraction (offline) ──────────────────────────
    print("[2/6] Extracting 22 URL-lexical features (offline mode) ...")
    X = np.array([extract_features(url, live_lookup=False) for url in df["url"]])
    y = df["label"].values
    print(f"      Feature matrix shape: {X.shape}")
    print()

    # ── Step 3: Simulate host-based + network features ────────────────
    print("[3/6] Simulating host-based & network features ...")
    #   10=domain_age_days, 11=whois_available, 12=dns_a_count, 13=dns_has_mx,
    #   14=is_shortened (already set), 15=redirect_count,
    #   16=ssl_valid, 17=ssl_days_remaining, 18=is_free_cert,
    #   19=tld_risk_score (already set), 20=path_depth (already set),
    #   21=has_punycode (already set)
    for i in range(len(X)):
        if y[i] == 0:  # Legitimate
            X[i][10] = float(random.randint(365, 9000))      # Old domain
            X[i][11] = 1.0                                    # WHOIS available
            X[i][12] = float(random.randint(1, 8))           # Multiple A records
            X[i][13] = 1.0                                    # MX record exists
            X[i][15] = float(random.choices([0, 1], weights=[0.8, 0.2])[0])
            X[i][16] = 1.0                                    # SSL valid
            X[i][17] = float(random.randint(30, 365))        # SSL days remaining
            X[i][18] = float(random.choices([0, 1], weights=[0.6, 0.4])[0])
        else:           # Phishing
            X[i][10] = float(random.choice([-1, *range(0, 90)]))
            X[i][11] = float(random.choices([0, 1], weights=[0.75, 0.25])[0])
            X[i][12] = float(random.choices([0, 0, 1], weights=[0.5, 0.3, 0.2])[0])
            X[i][13] = float(random.choices([0, 1], weights=[0.8, 0.2])[0])
            X[i][15] = float(random.choices([0, 1, 2, 3, 4],
                                             weights=[0.3, 0.2, 0.2, 0.2, 0.1])[0])
            X[i][16] = float(random.choices([0, 1], weights=[0.5, 0.5])[0])
            X[i][17] = float(random.choice([-1, *range(0, 30)]))
            X[i][18] = float(random.choices([1, 0], weights=[0.8, 0.2])[0])
    print("      Done.")
    print()

    # ── Step 4: Train / Test split ────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )
    print(f"[4/6] Train/Test split: {len(X_train)} train  |  {len(X_test)} test")
    print()

    # ── Step 5: Train XGBoost + Cross-Validation ──────────────────────
    print("[5/6] Training XGBoost classifier ...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("      Model trained successfully.")
    print()

    # Cross-validation
    print("      Running 5-fold cross-validation ...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"      CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print()

    # ── Step 6: Evaluate & Generate Reports ───────────────────────────
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, y_proba)

    print("[6/6] Evaluation Results")
    print("=" * 45)
    print(f"  Accuracy  : {accuracy:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC AUC   : {roc:.4f}")
    print("=" * 45)
    print()
    print("  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Legitimate", "Phishing"]))

    # Generate plots
    print("  Generating evaluation plots ...")
    _plot_confusion_matrix(
        y_test, y_pred,
        os.path.join(REPORTS_DIR, "confusion_matrix.png"))
    _plot_roc_curve(
        y_test, y_proba,
        os.path.join(REPORTS_DIR, "roc_curve.png"))
    _plot_feature_importance(
        model, FEATURE_NAMES,
        os.path.join(REPORTS_DIR, "feature_importance.png"))

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"\n[✓] Model saved to {MODEL_PATH}")
    print(f"[✓] Reports saved to {REPORTS_DIR}/")
    print()


if __name__ == "__main__":
    main()
