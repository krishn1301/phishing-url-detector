"""
utils.py - Feature Extraction & Safe Preview Module
=====================================================
Provides functions to extract numeric features from a URL for ML
classification and to safely preview a suspicious web page without
executing any scripts.
"""

import re
import math
import socket
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse
from collections import Counter

import requests
import whois
import dns.resolver
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUSPICIOUS_KEYWORDS = ["login", "verify", "secure", "bank", "update",
                       "account", "signin", "confirm", "password", "paypal"]

SPECIAL_CHARS = ["@", "-", "_"]

# Regex pattern to detect an IP address used as a hostname
IP_PATTERN = re.compile(
    r"^(?:https?://)?(\d{1,3}\.){3}\d{1,3}"
)

# Maximum time (seconds) we will wait when fetching a page for preview
PREVIEW_TIMEOUT = 5


# ===================================================================
# FEATURE EXTRACTION
# ===================================================================

def _shannon_entropy(text: str) -> float:
    """Calculate the Shannon entropy of a string."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum(
        (count / length) * math.log2(count / length)
        for count in counts.values()
    )


def _get_domain_age_days(hostname: str) -> float:
    """
    Query WHOIS for the domain and return its age in days.
    Returns -1.0 if the information is unavailable.
    """
    try:
        w = whois.whois(hostname)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation:
            age = (datetime.now(timezone.utc) - creation.replace(tzinfo=timezone.utc)).days
            return float(max(age, 0))
    except Exception:
        pass
    return -1.0


def _whois_available(hostname: str) -> int:
    """
    Return 1 if WHOIS data can be retrieved for the domain, else 0.
    A missing WHOIS record is a phishing indicator.
    """
    try:
        w = whois.whois(hostname)
        if w.domain_name:
            return 1
    except Exception:
        pass
    return 0


def _dns_a_record_count(hostname: str) -> int:
    """Return the number of DNS A records for *hostname*."""
    try:
        answers = dns.resolver.resolve(hostname, "A")
        return len(answers)
    except Exception:
        return 0


def _dns_has_mx(hostname: str) -> int:
    """Return 1 if the domain has at least one MX record, else 0."""
    try:
        answers = dns.resolver.resolve(hostname, "MX")
        return 1 if len(answers) > 0 else 0
    except Exception:
        return 0


def extract_features(url: str, live_lookup: bool = True) -> list:
    """
    Extract a numeric feature vector from *url*.

    Features (in order):
        0   url_length            – Total character length
        1   dot_count             – Number of '.' characters
        2   at_count              – Number of '@' characters
        3   dash_count            – Number of '-' characters
        4   underscore_count      – Number of '_' characters
        5   has_ip                – 1 if hostname is an IP address, else 0
        6   has_https             – 1 if scheme is HTTPS, else 0
        7   suspicious_keyword_count – Count of suspicious words found
        8   subdomain_count       – Number of subdomains (dots in hostname − 1)
        9   entropy               – Shannon entropy of the URL string
       10   domain_age_days       – Domain age in days (-1 if unknown)
       11   whois_available       – 1 if WHOIS data exists, else 0
       12   dns_a_record_count    – Number of DNS A records
       13   dns_has_mx            – 1 if MX record exists, else 0

    Args:
        url: The URL to analyse.
        live_lookup: If True, perform live WHOIS/DNS queries. Set to
                     False during training on synthetic data to avoid
                     thousands of network calls.

    Returns:
        list[float]: 14-element numeric vector.
    """
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
    except Exception:
        # If parsing fails, return a zero vector
        return [0.0] * 14

    hostname = parsed.hostname or ""

    # Basic counts
    url_length = len(url)
    dot_count = url.count(".")
    at_count = url.count("@")
    dash_count = url.count("-")
    underscore_count = url.count("_")

    # IP address check
    has_ip = 1 if IP_PATTERN.match(url) else 0

    # HTTPS check
    has_https = 1 if parsed.scheme == "https" else 0

    # Suspicious keywords
    url_lower = url.lower()
    suspicious_keyword_count = sum(
        1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower
    )

    # Subdomain count (dots in hostname minus 1, minimum 0)
    subdomain_count = max(hostname.count(".") - 1, 0)

    # Shannon entropy
    entropy = round(_shannon_entropy(url), 4)

    # --- Host-based features ---
    if live_lookup and hostname and not has_ip:
        domain_age_days = _get_domain_age_days(hostname)
        whois_avail = _whois_available(hostname)
        a_record_count = _dns_a_record_count(hostname)
        mx_exists = _dns_has_mx(hostname)
    elif live_lookup and has_ip:
        # IP-based URLs: no meaningful WHOIS/DNS for domain
        domain_age_days = -1.0
        whois_avail = 0
        a_record_count = 0
        mx_exists = 0
    else:
        # Synthetic / offline mode – will be set by the training script
        domain_age_days = -1.0
        whois_avail = 0
        a_record_count = 0
        mx_exists = 0

    return [
        float(url_length),
        float(dot_count),
        float(at_count),
        float(dash_count),
        float(underscore_count),
        float(has_ip),
        float(has_https),
        float(suspicious_keyword_count),
        float(subdomain_count),
        float(entropy),
        float(domain_age_days),
        float(whois_avail),
        float(a_record_count),
        float(mx_exists),
    ]


# Feature names matching the vector indices (used by the frontend)
FEATURE_NAMES = [
    "URL Length",
    "Dot Count",
    "@ Count",
    "Dash Count",
    "Underscore Count",
    "Has IP Address",
    "Has HTTPS",
    "Suspicious Keywords",
    "Subdomain Count",
    "Entropy",
    "Domain Age (days)",
    "WHOIS Available",
    "DNS A Records",
    "DNS Has MX",
]


# ===================================================================
# SAFE PREVIEW MODULE
# ===================================================================

def safe_preview(url: str) -> dict:
    """
    Safely fetch and analyse the content of *url*.

    The function:
      • Fetches raw HTML only (no JavaScript execution).
      • Parses with BeautifulSoup.
      • Extracts page title, form count, password field count,
        external form-action domains, and suspicious keyword presence.
      • Does NOT execute scripts or download files.

    Returns:
        dict with keys:
            page_title, form_count, password_fields,
            external_domains, suspicious_keywords_found, warnings
    """
    result = {
        "page_title": "N/A",
        "form_count": 0,
        "password_fields": 0,
        "external_domains": [],
        "suspicious_keywords_found": [],
        "warnings": [],
    }

    try:
        # Ensure URL has a scheme
        fetch_url = url if url.startswith(("http://", "https://")) else f"http://{url}"

        headers = {"User-Agent": "PhishingDetector/1.0 (Academic Research)"}
        response = requests.get(
            fetch_url, headers=headers, timeout=PREVIEW_TIMEOUT,
            allow_redirects=True, stream=False,
        )
        response.raise_for_status()

        # Only process HTML content
        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            result["warnings"].append("Response is not HTML content.")
            return result

        soup = BeautifulSoup(response.text, "html.parser")

        # --- Page title ---
        title_tag = soup.find("title")
        result["page_title"] = title_tag.get_text(strip=True) if title_tag else "No title"

        # --- Forms ---
        forms = soup.find_all("form")
        result["form_count"] = len(forms)

        # --- Password fields ---
        password_inputs = soup.find_all("input", attrs={"type": "password"})
        result["password_fields"] = len(password_inputs)

        # --- External form-action domains ---
        parsed_url = urlparse(fetch_url)
        page_domain = parsed_url.hostname or ""

        for form in forms:
            action = form.get("action", "")
            if action and action.startswith(("http://", "https://")):
                action_domain = urlparse(action).hostname or ""
                if action_domain and action_domain != page_domain:
                    result["external_domains"].append(action_domain)

        result["external_domains"] = list(set(result["external_domains"]))

        # --- Suspicious keywords in visible text ---
        page_text = soup.get_text(separator=" ").lower()
        found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in page_text]
        result["suspicious_keywords_found"] = found_keywords

        # --- Build warnings list ---
        if result["form_count"] > 0:
            result["warnings"].append(
                f"Page contains {result['form_count']} form(s)."
            )
        if result["password_fields"] > 0:
            result["warnings"].append(
                f"Page contains {result['password_fields']} password input field(s)."
            )
        if result["external_domains"]:
            result["warnings"].append(
                f"Forms submit data to external domain(s): "
                f"{', '.join(result['external_domains'])}"
            )
        if found_keywords:
            result["warnings"].append(
                f"Suspicious keywords detected: {', '.join(found_keywords)}"
            )

    except requests.RequestException as exc:
        logger.warning("Safe preview failed for %s: %s", url, exc)
        result["warnings"].append(f"Could not fetch page: {exc}")

    return result
