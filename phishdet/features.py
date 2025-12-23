"""Lexical feature extraction for URLs."""
from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Dict
from urllib.parse import urlparse

import numpy as np
import pandas as pd

_IPv4_RE = re.compile(
    r"^(?:\d{1,3}\.){3}\d{1,3}$"
)
_SUSPICIOUS_KEYWORDS = {
    "login",
    "signin",
    "verify",
    "update",
    "account",
    "secure",
    "bank",
    "paypal",
    "confirm",
    "password",
    "billing",
    "support",
}
_TOKEN_SPLIT_RE = re.compile(r"[\.-_/]+")


def _safe_parse(url: str):
    # Ensure scheme for urlparse consistency
    candidate = url if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url) else f"http://{url}"
    try:
        return urlparse(candidate)
    except Exception:
        return urlparse("http://")


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = np.bincount(np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8))
    probs = counts[counts > 0] / counts.sum()
    return float(-np.sum(probs * np.log2(probs))) if probs.size else 0.0


def extract_lexical_features(url: str) -> Dict[str, float]:
    """Extract deterministic lexical features from a URL string."""
    parsed = _safe_parse(url or "")
    full_url = parsed.geturl() or ""
    host = parsed.hostname or ""
    path = parsed.path or ""
    query = parsed.query or ""
    scheme = parsed.scheme or ""

    url_len = len(full_url)
    host_len = len(host)
    path_len = len(path)
    query_len = len(query)

    dot_count = full_url.count(".")
    dash_count = full_url.count("-")
    underscore_count = full_url.count("_")
    slash_count = full_url.count("/")
    at_count = full_url.count("@")
    qmark_count = full_url.count("?")
    eq_count = full_url.count("=")
    amp_count = full_url.count("&")
    percent_count = full_url.count("%")

    digit_count = sum(ch.isdigit() for ch in full_url)
    alpha_count = sum(ch.isalpha() for ch in full_url)

    has_https = 1.0 if scheme.lower() == "https" else 0.0
    has_ip_host = 1.0 if _IPv4_RE.match(host) else 0.0
    port_present = 1.0 if parsed.port else 0.0

    subdomain_count = float(max(host.count(".") - 1, 0)) if host else 0.0

    tokens = [t for t in _TOKEN_SPLIT_RE.split(host + "/" + path) if t]
    token_lengths = [len(t) for t in tokens]
    token_count = float(len(tokens))
    avg_token_len = float(np.mean(token_lengths)) if token_lengths else 0.0
    max_token_len = float(np.max(token_lengths)) if token_lengths else 0.0

    entropy = _shannon_entropy(full_url)

    lowered_tokens = [t.lower() for t in tokens]
    suspicious_hits = sum(1 for t in lowered_tokens if t in _SUSPICIOUS_KEYWORDS)

    return {
        "url_len": float(url_len),
        "host_len": float(host_len),
        "path_len": float(path_len),
        "query_len": float(query_len),
        "dot_count": float(dot_count),
        "dash_count": float(dash_count),
        "underscore_count": float(underscore_count),
        "slash_count": float(slash_count),
        "at_count": float(at_count),
        "qmark_count": float(qmark_count),
        "eq_count": float(eq_count),
        "amp_count": float(amp_count),
        "percent_count": float(percent_count),
        "digit_count": float(digit_count),
        "alpha_count": float(alpha_count),
        "has_https": has_https,
        "has_ip_host": has_ip_host,
        "port_present": port_present,
        "subdomain_count": subdomain_count,
        "token_count": token_count,
        "avg_token_len": avg_token_len,
        "max_token_len": max_token_len,
        "shannon_entropy": entropy,
        "suspicious_keyword_hits": float(suspicious_hits),
    }


def build_feature_df(urls: Iterable[str]) -> pd.DataFrame:
    records = [extract_lexical_features(u) for u in urls]
    df = pd.DataFrame.from_records(records)
    return df.fillna(0.0)
