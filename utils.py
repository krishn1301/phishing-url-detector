"""
utils.py - Enhanced Feature Extraction & Safe Preview Module
==============================================================
Provides functions to extract a 22-element numeric feature vector from
a URL for ML classification and to safely preview suspicious web pages.

Features cover URL-lexical, host-based, SSL/TLS, redirect-chain,
URL-shortener, and TLD-risk dimensions.
"""

import re
import ssl
import math
import socket
import logging
from datetime import datetime, timezone
from urllib.parse import urlparse
from collections import Counter

import requests
import whois
import dns.resolver
import tldextract
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUSPICIOUS_KEYWORDS = [
    "login", "verify", "secure", "bank", "update", "account", "signin",
    "confirm", "password", "paypal", "ebay", "apple", "microsoft",
    "netflix", "amazon", "wallet", "crypto", "suspend", "alert",
    "unusual", "locked", "expired", "urgent",
]

# Known URL shortener domains
URL_SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "ow.ly",
    "buff.ly", "rebrand.ly", "bl.ink", "short.io", "cutt.ly",
    "t.ly", "rb.gy", "v.gd", "tiny.cc", "shorturl.at", "clck.ru",
    "qps.ru", "lnkd.in", "db.tt", "soo.gd", "s2r.co",
}

# TLDs frequently abused in phishing campaigns
RISKY_TLDS = {
    "tk": 1.0, "ml": 1.0, "ga": 1.0, "cf": 1.0, "gq": 1.0,
    "xyz": 0.8, "top": 0.8, "work": 0.8, "click": 0.8, "loan": 0.9,
    "download": 0.7, "racing": 0.7, "win": 0.7, "bid": 0.7,
    "stream": 0.6, "gdn": 0.6, "icu": 0.7, "buzz": 0.6,
    "rest": 0.5, "fit": 0.5, "cam": 0.5, "surf": 0.5,
}

# Free / DV-only certificate issuers (common on phishing sites)
FREE_CERT_ISSUERS = [
    "let's encrypt", "letsencrypt", "zerossl", "buypass",
    "ssl.com", "cloudflare", "sectigo",  # free tier
]

# Regex pattern to detect an IP address used as a hostname
IP_PATTERN = re.compile(
    r"^(?:https?://)?\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
)

# Maximum time (seconds) for network operations
PREVIEW_TIMEOUT = 5
SSL_TIMEOUT = 4
REDIRECT_TIMEOUT = 5


# ===================================================================
# FEATURE EXTRACTION  (22 features)
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
    """Return 1 if WHOIS data can be retrieved for the domain, else 0."""
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


def _is_shortened(hostname: str) -> int:
    """Return 1 if the URL uses a known URL shortener service."""
    return 1 if hostname.lower() in URL_SHORTENERS else 0


def _count_redirects(url: str) -> int:
    """Follow the URL and return the number of redirects in the chain."""
    try:
        fetch = url if url.startswith(("http://", "https://")) else f"http://{url}"
        resp = requests.head(
            fetch, allow_redirects=True, timeout=REDIRECT_TIMEOUT,
            headers={"User-Agent": "PhishDetector/2.0"}
        )
        return len(resp.history)
    except Exception:
        return -1


def _ssl_info(hostname: str) -> tuple:
    """
    Fetch SSL certificate info for *hostname*.
    Returns (is_valid, days_remaining, is_free_cert).
    """
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(socket.socket(), server_hostname=hostname) as s:
            s.settimeout(SSL_TIMEOUT)
            s.connect((hostname, 443))
            cert = s.getpeercert()

        # Validity
        not_after = datetime.strptime(cert["notAfter"], "%b %d %H:%M:%S %Y %Z")
        days_remaining = (not_after - datetime.utcnow()).days
        is_valid = 1 if days_remaining > 0 else 0

        # Free certificate check
        issuer_str = str(cert.get("issuer", "")).lower()
        is_free = 1 if any(fi in issuer_str for fi in FREE_CERT_ISSUERS) else 0

        return (is_valid, float(max(days_remaining, 0)), is_free)
    except Exception:
        return (0, -1.0, 0)


def _tld_risk_score(hostname: str) -> float:
    """Return a risk score (0.0–1.0) based on the TLD."""
    ext = tldextract.extract(hostname)
    tld = ext.suffix.split(".")[-1].lower() if ext.suffix else ""
    return RISKY_TLDS.get(tld, 0.0)


def _path_depth(url: str) -> int:
    """Return the depth of the URL path (number of / segments)."""
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
        path = parsed.path.strip("/")
        return len(path.split("/")) if path else 0
    except Exception:
        return 0


def _has_punycode(hostname: str) -> int:
    """Return 1 if the hostname contains punycode (IDN homograph attack)."""
    return 1 if hostname.startswith("xn--") or ".xn--" in hostname else 0


def extract_features(url: str, live_lookup: bool = True) -> list:
    """
    Extract a 22-element numeric feature vector from *url*.

    Features (in order):
         0  url_length
         1  dot_count
         2  at_count
         3  dash_count
         4  underscore_count
         5  has_ip
         6  has_https
         7  suspicious_keyword_count
         8  subdomain_count
         9  entropy
        10  domain_age_days       (live)
        11  whois_available       (live)
        12  dns_a_record_count    (live)
        13  dns_has_mx            (live)
        14  is_shortened
        15  redirect_count        (live)
        16  ssl_valid             (live)
        17  ssl_days_remaining    (live)
        18  is_free_certificate   (live)
        19  tld_risk_score
        20  path_depth
        21  has_punycode
    """
    try:
        parsed = urlparse(url if "://" in url else f"http://{url}")
    except Exception:
        return [0.0] * 22

    hostname = parsed.hostname or ""

    # ---- URL-lexical features (always computed) ----
    url_length = len(url)
    dot_count = url.count(".")
    at_count = url.count("@")
    dash_count = url.count("-")
    underscore_count = url.count("_")
    has_ip = 1 if IP_PATTERN.match(url) else 0
    has_https = 1 if parsed.scheme == "https" else 0

    url_lower = url.lower()
    suspicious_keyword_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)
    subdomain_count = max(hostname.count(".") - 1, 0)
    entropy = round(_shannon_entropy(url), 4)

    # Features computable offline
    is_short = _is_shortened(hostname)
    tld_risk = _tld_risk_score(hostname)
    path_d = _path_depth(url)
    punycode = _has_punycode(hostname)

    # ---- Host-based / network features (live only) ----
    if live_lookup and hostname and not has_ip:
        domain_age_days = _get_domain_age_days(hostname)
        whois_avail = _whois_available(hostname)
        a_record_count = _dns_a_record_count(hostname)
        mx_exists = _dns_has_mx(hostname)
        redirect_count = _count_redirects(url)
        ssl_valid, ssl_days, is_free_cert = _ssl_info(hostname)
    elif live_lookup and has_ip:
        domain_age_days = -1.0
        whois_avail = 0
        a_record_count = 0
        mx_exists = 0
        redirect_count = _count_redirects(url)
        ssl_valid, ssl_days, is_free_cert = 0, -1.0, 0
    else:
        # Offline / training mode — caller will set these
        domain_age_days = -1.0
        whois_avail = 0
        a_record_count = 0
        mx_exists = 0
        redirect_count = 0
        ssl_valid, ssl_days, is_free_cert = 0, -1.0, 0

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
        float(is_short),
        float(redirect_count),
        float(ssl_valid),
        float(ssl_days),
        float(is_free_cert),
        float(tld_risk),
        float(path_d),
        float(punycode),
    ]


# Feature names matching the vector indices
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
    "Is Shortened URL",
    "Redirect Count",
    "SSL Valid",
    "SSL Days Remaining",
    "Free Certificate",
    "TLD Risk Score",
    "Path Depth",
    "Has Punycode",
]

# Grouping for the frontend
URL_FEATURES = FEATURE_NAMES[:10]
HOST_FEATURES = FEATURE_NAMES[10:14]
SECURITY_FEATURES = FEATURE_NAMES[14:]


# ===================================================================
# SAFE PREVIEW MODULE
# ===================================================================

def safe_preview(url: str) -> dict:
    """
    Safely fetch and analyse the content of *url*.

    Returns:
        dict with keys:
            page_title, form_count, password_fields,
            external_domains, suspicious_keywords_found,
            hidden_fields, iframe_count, warnings
    """
    result = {
        "page_title": "N/A",
        "form_count": 0,
        "password_fields": 0,
        "hidden_fields": 0,
        "iframe_count": 0,
        "external_domains": [],
        "suspicious_keywords_found": [],
        "warnings": [],
    }

    try:
        fetch_url = url if url.startswith(("http://", "https://")) else f"http://{url}"
        headers = {"User-Agent": "PhishingDetector/2.0 (Academic Research)"}
        response = requests.get(
            fetch_url, headers=headers, timeout=PREVIEW_TIMEOUT,
            allow_redirects=True, stream=False,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            result["warnings"].append("Response is not HTML content.")
            return result

        soup = BeautifulSoup(response.text, "html.parser")

        # Page title
        title_tag = soup.find("title")
        result["page_title"] = title_tag.get_text(strip=True) if title_tag else "No title"

        # Forms
        forms = soup.find_all("form")
        result["form_count"] = len(forms)

        # Password fields
        pw = soup.find_all("input", attrs={"type": "password"})
        result["password_fields"] = len(pw)

        # Hidden fields
        hidden = soup.find_all("input", attrs={"type": "hidden"})
        result["hidden_fields"] = len(hidden)

        # Iframes
        iframes = soup.find_all("iframe")
        result["iframe_count"] = len(iframes)

        # External form-action domains
        parsed_url = urlparse(fetch_url)
        page_domain = parsed_url.hostname or ""

        for form in forms:
            action = form.get("action", "")
            if action and action.startswith(("http://", "https://")):
                action_domain = urlparse(action).hostname or ""
                if action_domain and action_domain != page_domain:
                    result["external_domains"].append(action_domain)

        result["external_domains"] = list(set(result["external_domains"]))

        # Suspicious keywords in visible text
        page_text = soup.get_text(separator=" ").lower()
        found_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in page_text]
        result["suspicious_keywords_found"] = found_keywords

        # Build warnings
        if result["form_count"] > 0:
            result["warnings"].append(
                f"Page contains {result['form_count']} form(s)."
            )
        if result["password_fields"] > 0:
            result["warnings"].append(
                f"Page contains {result['password_fields']} password input field(s)."
            )
        if result["hidden_fields"] > 3:
            result["warnings"].append(
                f"Page contains {result['hidden_fields']} hidden input fields — potential data harvesting."
            )
        if result["iframe_count"] > 0:
            result["warnings"].append(
                f"Page embeds {result['iframe_count']} iframe(s) — may load external malicious content."
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
