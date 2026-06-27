import sys
sys.stdout.reconfigure(line_buffering=True)

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin, urlparse
import warnings
import threading
import time
import logging
import pandas as pd
import secrets
import json
import math
import os, json, math, hashlib, random
import requests
from datetime import datetime
from urllib.parse import urlparse
from groq import Groq
from dotenv import load_dotenv
import os
from rank import (
    RankPredictor,
    generate_synthetic_data,
    load_from_mysql,
    explode_results,
)
import threading

load_dotenv()  # ← loads .env into environment
warnings.filterwarnings("ignore", message="Unverified HTTPS request")

app = Flask(__name__, static_folder=".")
CORS(app)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "3306"))
DB_NAME = os.getenv("DB_NAME")
DB_AVAILABLE = bool(DB_USER and DB_HOST and DB_NAME)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")
if DB_AVAILABLE:
    app.config["SQLALCHEMY_DATABASE_URI"] = (
        f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        "?charset=utf8mb4"
    )
else:
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///fallback.db"
    print("⚠️  No DB_* env vars set — running with local SQLite fallback (no MySQL).", flush=True)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
if DB_AVAILABLE:
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 1800,  # recycle connections every 30 min
        "pool_pre_ping": True,  # test connection health before using
        "pool_size": 10,  # connection pool size
        "max_overflow": 20,  # extra connections above pool_size
        "connect_args": {
            "connect_timeout": 10,
        },
    }

db = SQLAlchemy(app)

analysis_status = {}


# ==================== GOOGLE CSE CONFIG ====================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # ← paste your Google API key
GOOGLE_CX = os.getenv("GOOGLE_CX")  # ← paste your Custom Search Engine ID


# ==================== SOFTWARE CAPABILITY TABLE ====================
class _Partial:
    def __bool__(self):
        return False

    def __repr__(self):
        return "Partial"


Partial = _Partial()

SOFTWARE_CAPABILITIES = {
    "ChiroTouch": {
        "online_booking": True,
        "digital_intake": True,
        "checkin": True,
        "telehealth": False,
        "patient_portal": True,
        "recalls": True,
    },
    "ChiroSpring": {
        "online_booking": True,
        "digital_intake": True,
        "checkin": True,
        "telehealth": False,
        "patient_portal": True,
        "recalls": True,
    },
    "ChiroFusion": {
        "online_booking": True,
        "digital_intake": True,
        "checkin": True,
        "telehealth": False,
        "patient_portal": True,
        "recalls": True,
    },
    "ChiroHD": {
        "online_booking": True,
        "digital_intake": True,
        "checkin": True,
        "telehealth": False,
        "patient_portal": True,
        "recalls": False,
    },
    "Eclipse EHR": {
        "online_booking": False,
        "digital_intake": Partial,
        "checkin": False,
        "telehealth": False,
        "patient_portal": False,
        "recalls": False,
    },
    "Genesis Chiropractic Software/Clinic Mind": {
        "online_booking": True,
        "digital_intake": True,
        "checkin": True,
        "telehealth": False,
        "patient_portal": True,
        "recalls": True,
    },
    "Jane App": {
        "online_booking": True,
        "digital_intake": True,
        "checkin": True,
        "telehealth": True,
        "patient_portal": True,
        "recalls": True,
    },
    "ReviewWave": {
        "online_booking": True,
        "digital_intake": True,
        "checkin": False,
        "telehealth": False,
        "patient_portal": False,
        "recalls": True,
    },
    "Kareo": {
        "online_booking": True,
        "digital_intake": True,
        "checkin": Partial,
        "telehealth": True,
        "patient_portal": True,
        "recalls": False,
    },
    "AdvancedMD": {
        "online_booking": True,
        "digital_intake": True,
        "checkin": True,
        "telehealth": True,
        "patient_portal": True,
        "recalls": True,
    },
    "Athenahealth": {
        "online_booking": True,
        "digital_intake": True,
        "checkin": True,
        "telehealth": True,
        "patient_portal": True,
        "recalls": True,
    },
    "Practice Fusion": {
        "online_booking": False,
        "digital_intake": Partial,
        "checkin": False,
        "telehealth": False,
        "patient_portal": True,
        "recalls": False,
    },
    "DrChrono": {
        "online_booking": True,
        "digital_intake": True,
        "checkin": True,
        "telehealth": True,
        "patient_portal": True,
        "recalls": False,
    },
    "Netsmart": {
        "online_booking": Partial,
        "digital_intake": True,
        "checkin": False,
        "telehealth": True,
        "patient_portal": True,
        "recalls": False,
    },
    "Other": {
        "online_booking": None,
        "digital_intake": None,
        "checkin": None,
        "telehealth": None,
        "patient_portal": None,
        "recalls": None,
    },
    "None": {
        "online_booking": False,
        "digital_intake": False,
        "checkin": False,
        "telehealth": False,
        "patient_portal": False,
        "recalls": False,
    },
}


def get_software_capability(software_name, feature):
    if not software_name:
        return None
    cap = SOFTWARE_CAPABILITIES.get(software_name, {})
    if not cap:
        sw_lower = software_name.lower()
        for key, data in SOFTWARE_CAPABILITIES.items():
            if key.lower() in sw_lower or sw_lower in key.lower():
                cap = data
                break 
    return cap.get(feature, None) 

class ConfidenceResult:
    def __init__(self, passed: bool, confidence: float, sources: list):
        self.passed = passed
        self.confidence = round(min(max(confidence, 0.0), 1.0), 2)
        self.sources = sources

    def to_dict(self):
        return {
            "passed": self.passed,
            "confidence": self.confidence,
            "sources": self.sources,
        }

    def __bool__(self):
        return self.passed


def _combine_confidence(scraper_signal, software_signal, groq_signal=None):
    signals = {}
    if scraper_signal is not None:
        signals["scraper"] = (scraper_signal, 0.40)
    if software_signal is not None:
        signals["software"] = (software_signal, 0.35)
    if groq_signal is not None:
        signals["groq"] = (groq_signal, 0.25)
    if not signals:
        return ConfidenceResult(False, 0.0, [])
    total_weight = sum(w for _, w in signals.values())
    weighted_sum = sum((1.0 if val else 0.0) * w for val, w in signals.values())
    confidence = weighted_sum / total_weight
    passed = confidence >= 0.55
    return ConfidenceResult(passed, confidence, list(signals.keys()))


def check_groq_available():
    return bool(GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here")


def analyze_with_groq(scraped_text, website_url, software=None):
    software_ctx = (
        f"The practice uses '{software}' as their practice-management software."
        if software
        else "The software they use is unknown."
    )
    print(f"\n  🤖 Sending content to Groq ({GROQ_MODEL})...", flush=True)
    try:
        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert healthcare/medical practice website analyzer. Always respond with valid JSON only. No explanations, no markdown, no extra text before or after the JSON.",
                },
                {
                    "role": "user",
                    "content": f"""Analyze this healthcare website and return ONLY valid JSON.

URL: {website_url}
Software context: {software_ctx}

Website Content:
{scraped_text[:3000]}

Return exactly this JSON structure with true/false values:
{{
  "mobile_responsive": true,
  "has_online_booking": false,
  "has_digital_intake": false,
  "has_patient_forms": false,
  "has_checkin": false,
  "has_testimonials": false,
  "has_blog": false,
  "has_social_media": false,
  "has_video": false,
  "has_contact_info": true,
  "has_ssl": true,
  "has_google_maps": false,
  "overall_quality": "average",
  "improvement_suggestions": ["suggestion1", "suggestion2"],
  "summary": "One sentence summary of the website quality."
}}

Rules:
- overall_quality must be: poor / average / good / excellent
- Set mobile_responsive=true if you see viewport meta, CSS frameworks, or @media queries
- Set has_digital_intake=true if you see ANY online form, iframe form widget, or patient portal link
- Set has_video=true if you see YouTube/Vimeo iframes, <video> tags, or any video player embed
- JSON only. Nothing else""",
                },
            ],
            temperature=0.1,
            max_tokens=600,
        )
        raw_text = response.choices[0].message.content
        print(f"Groq raw response: {raw_text[:200]}", flush=True)
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            print("Groq analysis successful", flush=True)
            return result
        return None
    except Exception as e:
        print(f"  ⚠️  Groq error: {e}", flush=True)
        return None


# ==================== OTP STORAGE (in-memory) ====================
otp_storage = {}


def generate_otp(length=6):
    return "".join([str(secrets.randbelow(10)) for _ in range(length)])


def cleanup_expired_otps():
    current_time = datetime.utcnow()
    expired = [
        phone
        for phone, data in otp_storage.items()
        if current_time > data["expires_at"]
    ]
    for phone in expired:
        del otp_storage[phone]


# ==================== DATABASE MODELS ====================
# NOTE: MySQL-specific changes:
#   - String lengths explicitly set (MySQL requires them for indexed columns)
#   - Text type used for long fields
#   - utf8mb4 charset handled at engine level


class Lead(db.Model):
    __tablename__ = "leads"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    practice_name = db.Column(db.String(200), nullable=False, index=True)
    website_url = db.Column(db.String(500), nullable=False)
    provider_name = db.Column(db.String(200), nullable=False)
    email = db.Column(db.String(200), nullable=False, index=True)
    phone = db.Column(db.String(50))
    zip_code = db.Column(db.String(20))
    current_software = db.Column(db.String(200))
    score = db.Column(db.Integer, index=True)
    seo_score = db.Column(db.Integer)
    engagement_score = db.Column(db.Integer)
    ai_summary = db.Column(db.Text)
    ai_suggestions = db.Column(db.Text)
    analysis_method = db.Column(db.String(50))
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    # Relationships
    analysis_results = db.relationship(
        "AnalysisResult", backref="lead", lazy="dynamic", cascade="all, delete-orphan"
    )
    corrections = db.relationship(
        "CorrectionLog", backref="lead", lazy="dynamic", cascade="all, delete-orphan"
    )
    rank_checks = db.relationship(
        "RankCheck", backref="lead", lazy="dynamic", cascade="all, delete-orphan"
    )

    def to_dict(self):
        return {
            "id": self.id,
            "practice_name": self.practice_name,
            "website_url": self.website_url,
            "provider_name": self.provider_name,
            "email": self.email,
            "phone": self.phone,
            "zip_code": self.zip_code,
            "current_software": self.current_software,
            "score": self.score,
            "seo_score": self.seo_score,
            "engagement_score": self.engagement_score,
            "ai_summary": self.ai_summary,
            "ai_suggestions": self.ai_suggestions,
            "analysis_method": self.analysis_method,
            "created_at": self.created_at.isoformat(),
        }


class AnalysisResult(db.Model):
    __tablename__ = "analysis_results"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    lead_id = db.Column(
        db.Integer,
        db.ForeignKey("leads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    criterion_name = db.Column(db.String(200), nullable=False)
    passed = db.Column(db.Boolean, nullable=False)
    confidence = db.Column(db.Float, default=0.0)
    sources = db.Column(db.String(200))
    points = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class CorrectionLog(db.Model):
    __tablename__ = "correction_logs"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    lead_id = db.Column(
        db.Integer,
        db.ForeignKey("leads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    criterion_name = db.Column(db.String(200), nullable=False)
    ai_decision = db.Column(db.Boolean, nullable=False)
    human_decision = db.Column(db.Boolean, nullable=False)
    ai_confidence = db.Column(db.Float, default=0.0)
    reason = db.Column(db.Text)
    corrected_by = db.Column(db.String(200))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "lead_id": self.lead_id,
            "criterion_name": self.criterion_name,
            "ai_decision": self.ai_decision,
            "human_decision": self.human_decision,
            "ai_confidence": self.ai_confidence,
            "reason": self.reason,
            "corrected_by": self.corrected_by,
            "created_at": self.created_at.isoformat(),
        }


class RankCheck(db.Model):
    """New table — stores ranking check history per lead."""

    __tablename__ = "rank_checks"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    lead_id = db.Column(
        db.Integer,
        db.ForeignKey("leads.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    domain = db.Column(db.String(300), nullable=False, index=True)
    location = db.Column(db.String(200))
    device = db.Column(db.String(20), default="mobile")
    keywords_json = db.Column(db.Text)  # JSON array of keywords
    results_json = db.Column(db.Text)  # JSON array of rank results
    metrics_json = db.Column(db.Text)  # JSON object of visibility metrics
    visibility_score = db.Column(db.Integer)
    avg_rank = db.Column(db.Float)
    live_data = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "lead_id": self.lead_id,
            "domain": self.domain,
            "location": self.location,
            "device": self.device,
            "keywords": json.loads(self.keywords_json or "[]"),
            "results": json.loads(self.results_json or "[]"),
            "metrics": json.loads(self.metrics_json or "{}"),
            "visibility_score": self.visibility_score,
            "avg_rank": self.avg_rank,
            "live_data": self.live_data,
            "created_at": self.created_at.isoformat(),
        }


PAGESPEED_API_KEY = os.getenv("PAGESPEED_API_KEY", "")


# ==================== PAGESPEED ====================
def get_google_pagespeed_scores(url):
    results = {
        "desktop": {"score": 0, "has_data": False, "error": None},
        "mobile": {"score": 0, "has_data": False, "error": None},
    }
    try:
        for strategy in ["desktop", "mobile"]:
            print(f"  Fetching Google PageSpeed score for {strategy}...", flush=True)
            api_url = "https://www.googleapis.com/pagespeedonline/v5/runPagespeed"
            params = {"url": url, "strategy": strategy, "category": "PERFORMANCE"}
            if PAGESPEED_API_KEY:
                params["key"] = PAGESPEED_API_KEY
            resp = requests.get(api_url, params=params, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                score = (
                    data.get("lighthouseResult", {})
                    .get("categories", {})
                    .get("performance", {})
                    .get("score")
                )
                if score is not None:
                    results[strategy]["score"] = int(score * 100)
                    results[strategy]["has_data"] = True
                    print(
                        f"    ✓ {strategy.capitalize()} score: {results[strategy]['score']}/100",
                        flush=True,
                    )
            else:
                print(f"    ✗ PageSpeed API error {resp.status_code}", flush=True)
    except Exception as e:
        print(f"PageSpeed error: {e}", flush=True)
    return results


# ==================== RANK CHECKER ====================
def get_rank_via_serpapi(
    keyword: str, domain: str, location: str = "India", device: str = "mobile"
) -> dict:
    """
    Fetch real Google rankings using SerpApi.
    Returns organic rank, local pack rank, and featured snippet detection.
    """
    if not SERPAPI_KEY:
        return estimate_rank_fallback(keyword, domain)

    try:
        # Map device string to SerpApi parameter
        device_param = "mobile" if device == "mobile" else "desktop"

        params = {
            "engine": "google",
            "q": keyword,
            "location": location,  # e.g. "Austin, Texas, United States" or "India"
            "hl": "en",
            "gl": "in",  # country code — change to "us" for US clients
            "num": 100,  # fetch up to 100 results (organic)
            "device": device_param,
            "api_key": SERPAPI_KEY,
            "no_cache": False,  # use SerpApi cache to save credits
        }

        resp = requests.get(
            "https://serpapi.com/search",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        domain_clean = domain.lower().replace("www.", "").strip("/")

        result = {
            "keyword": keyword,
            "rank": None,
            "local_rank": None,
            "featured": False,
            "url": None,
            "title": None,
            "found": False,
            "source": "serpapi",
            "device": device,
            "location": location,
        }

        # ── 1. Check featured snippet ──
        answer_box = data.get("answer_box", {})
        if answer_box:
            link = answer_box.get("link", "") or answer_box.get("displayed_link", "")
            if domain_clean in link.lower():
                result.update(
                    {
                        "rank": 0,
                        "found": True,
                        "featured": True,
                        "url": link,
                        "title": answer_box.get("title", "Featured Snippet"),
                    }
                )
                return result

        # ── 2. Check organic results (positions 1–100) ──
        organic = data.get("organic_results", [])
        for idx, item in enumerate(organic, start=1):
            link = item.get("link", "")
            if domain_clean in link.lower():
                result.update(
                    {
                        "rank": idx,
                        "found": True,
                        "url": link,
                        "title": item.get("title", ""),
                    }
                )
                break

        # ── 3. Check local pack (Google Maps 3-pack) ──
        local_results = data.get("local_results", {})
        if isinstance(local_results, dict):
            places = local_results.get("places", [])
        else:
            places = local_results or []

        for idx, place in enumerate(places, start=1):
            website = (
                place.get("website") or place.get("links", {}).get("website") or ""
            ).lower()
            title = (place.get("title") or place.get("name") or "").lower()
            if domain_clean in website or domain_clean in title:
                result["local_rank"] = idx
                if not result["found"]:  # local-only presence
                    result.update(
                        {"found": True, "title": place.get("title", ""), "url": website}
                    )
                break

        if not result["found"]:
            result["note"] = "Not in top 100 organic or local pack"

        return result

    except requests.exceptions.Timeout:
        print(f"SerpApi timeout for '{keyword}'", flush=True)
        return estimate_rank_fallback(keyword, domain)
    except Exception as e:
        print(f"SerpApi error for '{keyword}': {e}", flush=True)
        return estimate_rank_fallback(keyword, domain)


def estimate_rank_fallback(keyword: str, domain: str) -> dict:
    """Deterministic estimated rank — used when SerpApi key is absent."""
    seed = int(hashlib.md5((domain + keyword).encode()).hexdigest(), 16) % 1000
    random.seed(seed)
    rank = random.randint(8, 55)
    return {
        "keyword": keyword,
        "rank": rank,
        "local_rank": None,
        "featured": False,
        "found": True,
        "source": "estimated",
        "note": "Estimated rank — add SERPAPI_KEY to .env for live data",
    }


def calculate_visibility_score(rank_results: list) -> dict:
    """
    Compute visibility metrics with local pack bonus.
    Local pack positions get a CTR bonus because they appear above organic.
    """
    # Organic CTR curve (Backlinko 2023 data)
    CTR_CURVE = {
        0: 28.5,  # Featured snippet
        1: 27.6,
        2: 15.8,
        3: 11.0,
        4: 8.4,
        5: 6.3,
        6: 4.9,
        7: 3.9,
        8: 3.3,
        9: 2.7,
        10: 2.4,
    }
    # Local pack CTR (positions 1–3 in maps pack)
    LOCAL_CTR = {1: 14.0, 2: 7.0, 3: 4.5}

    total = len(rank_results)
    if total == 0:
        return {
            "visibility_score": 0,
            "avg_rank": None,
            "top_3": 0,
            "top_10": 0,
            "top_30": 0,
            "not_ranking": 0,
            "total_keywords": 0,
            "ctr_estimate": 0,
            "local_pack_count": 0,
            "featured_count": 0,
        }

    total_ctr = 0
    ranks = []
    top3 = top10 = top30 = 0
    local_count = 0
    featured_count = 0

    for r in rank_results:
        rank = r.get("rank")
        local_rank = r.get("local_rank")
        featured = r.get("featured", False)

        # Organic CTR
        if rank is not None:
            ctr = CTR_CURVE.get(rank, max(0.3, 2.4 - (rank - 10) * 0.06))
            total_ctr += ctr
            ranks.append(rank)
            if rank <= 3:
                top3 += 1
            if rank <= 10:
                top10 += 1
            if rank <= 30:
                top30 += 1
        else:
            total_ctr += 0  # not ranking

        # Local pack bonus
        if local_rank:
            total_ctr += LOCAL_CTR.get(local_rank, 2.0)
            local_count += 1

        if featured:
            featured_count += 1

    avg_ctr = total_ctr / total
    visibility_score = round(min(100, avg_ctr * 2.5))
    avg_rank = round(sum(ranks) / len(ranks), 1) if ranks else None
    not_ranking = sum(1 for r in rank_results if not r.get("rank"))

    return {
        "visibility_score": visibility_score,
        "avg_rank": avg_rank,
        "top_3": top3,
        "top_10": top10,
        "top_30": top30,
        "not_ranking": not_ranking,
        "total_keywords": total,
        "ctr_estimate": round(avg_ctr, 2),
        "local_pack_count": local_count,
        "featured_count": featured_count,
    }


# ==================== MULTI-PAGE ANALYZER ====================
class MultiPageAnalyzer:
    def __init__(self, url, max_pages=5, practice_name=None, software=None):
        self.base_url = url
        self.domain = urlparse(url).netloc
        self.max_pages = max_pages
        self.pages_content = {}
        self.visited_urls = set()
        self.practice_name_hint = practice_name
        self.software = software

    def fetch_page(self, url, timeout=10):
        from curl_cffi import requests as cf_requests

        urls_to_try = [url]
        if url.startswith("https://"):
            urls_to_try.append(url.replace("https://", "http://"))

        for attempt_url in urls_to_try:
            try:
                response = cf_requests.get(
                    attempt_url,
                    impersonate="chrome120",  # mimics Chrome 120 TLS fingerprint
                    timeout=timeout,
                    allow_redirects=True,
                )
                if response.status_code == 403:
                    print(f"  ✗ 403 still blocked after impersonation: {attempt_url}", flush=True)
                    return None
                response.raise_for_status()
                return response.text
            except Exception as e:
                print(f"  ✗ Error fetching {attempt_url}: {str(e)[:80]}", flush=True)
                continue

        return None

    def is_same_domain(self, url):
        try:
            parsed = urlparse(url)
            return (
                parsed.netloc == self.domain
                or parsed.netloc == f"www.{self.domain}"
                or parsed.netloc == self.domain.replace("www.", "")
            )
        except:
            return False

    def extract_links(self, html, current_url):
        soup = BeautifulSoup(html, "html.parser")
        links = set()
        for a_tag in soup.find_all("a", href=True):
            href = a_tag["href"]
            absolute_url = urljoin(current_url, href)
            clean_url = absolute_url.split("#")[0].split("?")[0]
            if clean_url.endswith("/"):
                clean_url = clean_url[:-1]
            if self.is_same_domain(clean_url) and clean_url not in self.visited_urls:
                links.add(clean_url)
        return links

    def crawl(self):
        print(f"\n{'='*60}", flush=True)
        print(f"Crawling: {self.base_url}", flush=True)
        print(f"{'='*60}", flush=True)
        priority_keywords = [
            "about",
            "services",
            "contact",
            "team",
            "staff",
            "appointment",
            "book",
            "schedule",
            "patient",
            "forms",
            "intake",
            "blog",
            "articles",
            "testimonial",
            "review",
        ]
        to_visit = [self.base_url]
        priority_queue = []
        while len(self.visited_urls) < self.max_pages and (to_visit or priority_queue):
            current_url = priority_queue.pop(0) if priority_queue else to_visit.pop(0)
            if current_url in self.visited_urls:
                continue
            print(
                f"Fetching page {len(self.visited_urls)+1}/{self.max_pages}: {current_url[:80]}",
                flush=True,
            )
            html = self.fetch_page(current_url)
            if html:
                self.visited_urls.add(current_url)
                self.pages_content[current_url] = html
                new_links = self.extract_links(html, current_url)
                for link in new_links:
                    if link not in self.visited_urls:
                        if any(kw in link.lower() for kw in priority_keywords):
                            if link not in priority_queue:
                                priority_queue.append(link)
                        else:
                            if link not in to_visit:
                                to_visit.append(link)
        print(f"\n  ✓ Crawled {len(self.visited_urls)} pages", flush=True)
        return len(self.visited_urls) > 0

    def search_all_pages(self, keywords):
        for url, html in self.pages_content.items():
            text = BeautifulSoup(html, "html.parser").get_text().lower()
            for keyword in keywords:
                if keyword in text:
                    return True
        return False

    def _html_contains_any(self, patterns):
        for html in self.pages_content.values():
            hl = html.lower()
            if any(p in hl for p in patterns):
                return True
        return False

    def check_mobile_responsive(self, groq_result=None) -> ConfidenceResult:
        scraper = False
        signals = []
        
        for html in self.pages_content.values():
            soup = BeautifulSoup(html, 'html.parser')
            html_lower = html.lower()
            
            # 1. Viewport meta tag (strongest signal)
            if soup.find('meta', attrs={'name': 'viewport'}):
                signals.append('viewport_meta')
                scraper = True; break
            
            # 2. CSS frameworks
            if any(fw in html_lower for fw in [
                'bootstrap', 'foundation', 'tailwind', 'bulma', 'materialize',
                'semantic-ui', 'uikit', 'skeleton'
            ]):
                signals.append('css_framework')
                scraper = True; break
            
            # 3. CSS media queries in <style> tags or linked CSS
            style_tags = soup.find_all('style')
            for style in style_tags:
                if style.string and '@media' in style.string:
                    signals.append('media_query_style_tag')
                    scraper = True; break
            if scraper: break
            
            # 4. Responsive meta in linked stylesheets or inline style attrs
            if 'max-width' in html_lower and 'min-width' in html_lower:
                signals.append('responsive_css_hints')
                scraper = True; break
            
            # 5. Known responsive page builders / themes
            if any(builder in html_lower for builder in [
                'elementor', 'divi', 'beaver builder', 'wpbakery',
                'squarespace', 'wix', 'webflow', 'shopify', 'framer'
            ]):
                signals.append('page_builder')
                scraper = True; break
            
            # 6. Responsive image tags
            if soup.find('img', attrs={'srcset': True}) or soup.find('picture'):
                signals.append('responsive_images')
                scraper = True; break
        
        # 7. Run Google PageSpeed Mobile — if score > 50, it's mobile-friendly
        try:
            ps = get_google_pagespeed_scores(self.base_url)
            mobile_score = ps.get('mobile', {}).get('score', 0)
            if mobile_score >= 60:
                signals.append('pagespeed_mobile_pass')
                scraper = True
            elif mobile_score > 0:
                # PageSpeed returned a real score but it's low — override scraper
                signals.append('pagespeed_mobile_fail')
                scraper = False  # low mobile score = NOT mobile responsive
        except:
            pass
        
        groq_val = bool(groq_result.get('mobile_responsive')) if groq_result else None
        print(f"  📱 Mobile signals: {signals}", flush=True)
        return _combine_confidence(scraper, None, groq_val)

    def check_online_booking(self, groq_result=None) -> ConfidenceResult:
        keyword_signal = self.search_all_pages(
            [
                "book appointment",
                "schedule now",
                "online booking",
                "book online",
                "request appointment",
                "schedule appointment",
            ]
        )
        widget_signal = self._html_contains_any(
            [
                "janeapp",
                "schedulicity",
                "calendly",
                "acuity",
                "zocdoc",
                "healthgrades",
                "setmore",
                "simplepractice",
            ]
        )
        scraper = keyword_signal or widget_signal
        software_val = get_software_capability(self.software, "online_booking")
        sw_bool = bool(software_val) if software_val is not None else None
        groq_val = bool(groq_result.get("has_online_booking")) if groq_result else None
        result = _combine_confidence(scraper, sw_bool, groq_val)
        _log_check("Online appointment booking", result)
        return result

    def check_digital_intake(self, groq_result=None) -> ConfidenceResult:
        scraper = False
        signals = []
        
        for html in self.pages_content.values():
            soup = BeautifulSoup(html, 'html.parser')
            html_lower = html.lower()
            text_lower = soup.get_text().lower()
            
            # 1. Text keywords (expanded list)
            intake_keywords = [
                'online intake', 'digital intake', 'fill forms online',
                'complete your forms online', 'paperless intake', 'electronic intake',
                'online forms', 'fill out forms', 'patient portal', 'complete forms',
                'new patient forms online', 'intake form online', 'registration online',
                'fill in your details', 'pre-registration', 'pre registration',
            ]
            if any(kw in text_lower for kw in intake_keywords):
                signals.append('text_keyword')
                scraper = True; break
            
            # 2. Known intake / form platforms in iframes or scripts
            intake_platforms = [
                'intakeq.com', 'jotform.com', 'typeform.com', 'formstack.com',
                'cognito', 'wufoo.com', 'paperform.co', 'gravity forms',
                'wpforms', 'caldera', 'ninja forms', 'formidable',
                'healthie.com', 'drchrono.com', 'simplepractice.com',
                'janeapp.com', 'kareo.com', 'nuvolo', 'phreesia.com',
                'docusign.com', 'hellosign.com', 'pandadoc.com',
            ]
            for iframe in soup.find_all('iframe'):
                src = iframe.get('src', '').lower()
                if any(platform in src for platform in intake_platforms):
                    signals.append(f'intake_iframe:{src[:40]}')
                    scraper = True; break
            if scraper: break
            
            # 3. Generic iframe containing "form" or "intake" in src
            for iframe in soup.find_all('iframe'):
                src = iframe.get('src', '').lower()
                if any(kw in src for kw in ['intake', 'form', 'register', 'patient']):
                    signals.append('generic_form_iframe')
                    scraper = True; break
            if scraper: break
            
            # 4. Online form elements on the page itself (input fields in forms)
            forms = soup.find_all('form')
            for form in forms:
                form_text = form.get_text().lower()
                if any(kw in form_text for kw in [
                    'date of birth', 'dob', 'insurance', 'chief complaint',
                    'medical history', 'emergency contact', 'referring physician'
                ]):
                    signals.append('inline_medical_form')
                    scraper = True; break
            if scraper: break
            
            # 5. Links to external intake forms
            for a_tag in soup.find_all('a', href=True):
                href = a_tag.get('href', '').lower()
                link_text = a_tag.get_text().lower()
                if any(platform in href for platform in intake_platforms):
                    signals.append('intake_link')
                    scraper = True; break
                if any(kw in link_text for kw in ['intake form', 'new patient form', 'patient intake']):
                    signals.append('intake_link_text')
                    scraper = True; break
            if scraper: break
        
        software_val = get_software_capability(self.software, 'digital_intake')
        sw_bool = True if software_val is True else (
            False if software_val is False or isinstance(software_val, _Partial) else None
        )
        groq_val = bool(groq_result.get('has_digital_intake')) if groq_result else None
        
        print(f"  📋 Intake signals: {signals}", flush=True)
        result = _combine_confidence(scraper, sw_bool, groq_val)
        _log_check("Digital patient intake (online)", result)
        return result

    def check_patient_forms(self, groq_result=None) -> ConfidenceResult:
        scraper = self.search_all_pages(
            [
                "patient intake",
                "new patient form",
                "patient form",
                "intake form",
                "registration form",
                "patient registration",
                "new patient",
                "patient paperwork",
                "download form",
            ]
        )
        software_val = get_software_capability(self.software, "digital_intake")
        sw_bool = bool(software_val) if software_val is not None else None
        groq_val = bool(groq_result.get("has_patient_forms")) if groq_result else None
        result = _combine_confidence(scraper, sw_bool, groq_val)
        _log_check("Patient forms available", result)
        return result

    def check_checkin(self, groq_result=None) -> ConfidenceResult:
        scraper = self.search_all_pages(
            ["check in", "check-in", "kiosk", "self check", "arrival"]
        )
        software_val = get_software_capability(self.software, "checkin")
        sw_bool = bool(software_val) if software_val is not None else None
        groq_val = bool(groq_result.get("has_checkin")) if groq_result else None
        result = _combine_confidence(scraper, sw_bool, groq_val)
        _log_check("Modern check-in system", result)
        return result

    def check_social_media(self, groq_result=None) -> ConfidenceResult:
        social_domains = [
            "facebook.com",
            "instagram.com",
            "twitter.com",
            "linkedin.com",
            "youtube.com",
            "tiktok.com",
        ]
        scraper = False
        for html in self.pages_content.values():
            soup = BeautifulSoup(html, "html.parser")
            for link in soup.find_all("a", href=True):
                if any(s in link["href"].lower() for s in social_domains):
                    scraper = True
                    break
            if scraper:
                break
        groq_val = bool(groq_result.get("has_social_media")) if groq_result else None
        result = _combine_confidence(scraper, None, groq_val)
        _log_check("Active social media presence", result)
        return result

    def check_testimonials(self, groq_result=None) -> ConfidenceResult:
        scraper = self.search_all_pages(
            [
                "testimonial",
                "review",
                "patient says",
                "success stor",
                "patient feedback",
                "what our patients",
                "patient stories",
            ]
        )
        groq_val = bool(groq_result.get("has_testimonials")) if groq_result else None
        result = _combine_confidence(scraper, None, groq_val)
        _log_check("Patient testimonials/reviews", result)
        return result

    def check_blog(self, groq_result=None) -> ConfidenceResult:
        url_signal = any(
            any(kw in url.lower() for kw in ["blog", "article", "news", "resource"])
            for url in self.pages_content.keys()
        )
        link_signal = False
        for html in self.pages_content.values():
            soup = BeautifulSoup(html, "html.parser")
            for link in soup.find_all("a", href=True):
                text = link.get_text().lower()
                href = link["href"].lower()
                if "blog" in text or "blog" in href or "article" in text:
                    link_signal = True
                    break
            if link_signal:
                break
        scraper = url_signal or link_signal
        groq_val = bool(groq_result.get("has_blog")) if groq_result else None
        result = _combine_confidence(scraper, None, groq_val)
        _log_check("Educational content/blog", result)
        return result


    def check_video(self, groq_result=None) -> ConfidenceResult:
        scraper = False
        signals = []

        for html in self.pages_content.values():
            soup = BeautifulSoup(html, "html.parser")
            html_lower = html.lower()
            text_lower = soup.get_text().lower()

            # 1. Native <video> tag
            if soup.find("video"):
                signals.append("html5_video_tag")
                scraper = True
                break

            # 2. Video platform iframes (expanded)
            video_platforms = [
                "youtube.com",
                "youtu.be",
                "vimeo.com",
                "loom.com",
                "wistia.com",
                "vzaar.com",
                "dailymotion.com",
                "brightcove.com",
                "vidyard.com",
                "sproutvideo.com",
                "jwplayer.com",
                "rumble.com",
                "bunnycdn.com",
            ]
            for iframe in soup.find_all("iframe"):
                src = iframe.get("src", "").lower()
                if any(p in src for p in video_platforms):
                    signals.append(f"video_iframe:{src[:40]}")
                    scraper = True
                    break
            if scraper:
                break

            # 3. Video platform links (YouTube/Vimeo linked but not embedded)
            for a_tag in soup.find_all("a", href=True):
                href = a_tag.get("href", "").lower()
                if any(p in href for p in ["youtube.com/watch", "youtu.be/", "vimeo.com/"]):
                    signals.append("video_link")
                    scraper = True
                    break
            if scraper:
                break

            # 4. Video JS players / SDKs in scripts
            video_js_patterns = [
                "jwplayer",
                "video.js",
                "videojs",
                "plyr.io",
                "mediaelement",
                "flowplayer",
                "wistia",
                "vidyard",
                "brightcove",
            ]
            script_tags = soup.find_all("script")
            for script in script_tags:
                src = script.get("src", "").lower()
                content = (script.string or "").lower()
                if any(p in src or p in content for p in video_js_patterns):
                    signals.append("video_js_player")
                    scraper = True
                    break
            if scraper:
                break

            # 5. Video thumbnail images (common YouTube thumbnail URLs)
            for img in soup.find_all("img"):
                src = img.get("src", "").lower()
                if "ytimg.com" in src or "img.youtube.com" in src or "vumbnail" in src:
                    signals.append("youtube_thumbnail")
                    scraper = True
                    break
            if scraper:
                break

            # 6. Keywords in page text suggesting video content
            video_keywords = [
                "watch our video",
                "watch video",
                "play video",
                "video tour",
                "see our video",
                "video about",
                "meet our team video",
                "virtual tour",
                "view our video",
            ]
            if any(kw in text_lower for kw in video_keywords):
                signals.append("video_text_keyword")
                scraper = True
                break

            # 7. og:video or twitter:player meta tags
            for meta in soup.find_all("meta"):
                prop = (meta.get("property") or meta.get("name") or "").lower()
                if "video" in prop:
                    signals.append("og_video_meta")
                    scraper = True
                    break
            if scraper:
                break

        groq_val = bool(groq_result.get("has_video")) if groq_result else None
        print(f"  🎥 Video signals: {signals}", flush=True)
        result = _combine_confidence(scraper, None, groq_val)
        _log_check("Video content about services", result)
        return result

    def check_ssl(self, groq_result=None) -> ConfidenceResult:
        scraper = self.base_url.startswith("https://")
        groq_val = bool(groq_result.get("has_ssl")) if groq_result else None
        result = _combine_confidence(scraper, None, groq_val)
        _log_check("SSL certificate (HTTPS)", result)
        return result

    def check_contact_info(self, groq_result=None) -> ConfidenceResult:
        phone_pattern = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        scraper = False
        for html in self.pages_content.values():
            if re.search(phone_pattern, html) or re.search(email_pattern, html):
                scraper = True
                break
        groq_val = bool(groq_result.get("has_contact_info")) if groq_result else None
        result = _combine_confidence(scraper, None, groq_val)
        _log_check("Contact information clearly visible", result)
        return result

    def check_google_maps(self, groq_result=None) -> ConfidenceResult:
        scraper = False
        for html in self.pages_content.values():
            soup = BeautifulSoup(html, "html.parser")
            for iframe in soup.find_all("iframe"):
                src = iframe.get("src", "").lower()
                if "google.com/maps" in src or "maps.google.com" in src:
                    scraper = True
                    break
            for link in soup.find_all("a", href=True):
                href = link["href"].lower()
                if any(
                    p in href
                    for p in [
                        "google.com/maps/place",
                        "maps.google.com",
                        "g.page/",
                        "goo.gl/maps",
                    ]
                ):
                    scraper = True
                    break
            if soup.find(attrs={"data-place-id": True}):
                scraper = True
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    schema = json.loads(script.string or "{}")
                    nodes = (
                        schema
                        if isinstance(schema, list)
                        else schema.get("@graph", [schema])
                    )
                    for node in nodes:
                        t = str(node.get("@type", "")).lower()
                        if any(
                            x in t
                            for x in [
                                "localbusiness",
                                "medicalbusiness",
                                "physician",
                                "chiropractor",
                            ]
                        ):
                            scraper = True
                            break
                except:
                    pass
            page_text = soup.get_text().lower()
            if any(
                p in page_text
                for p in ["google reviews", "google rating", "google business"]
            ):
                scraper = True
            if scraper:
                break

        nominatim_found = False
        if not scraper and self.practice_name_hint:
            try:
                params = {
                    "q": self.practice_name_hint,
                    "format": "json",
                    "limit": 3,
                    "addressdetails": 1,
                    "countrycodes": "us",
                }
                headers = {"User-Agent": "RankPulse -GMB-Checker/1.0"}
                resp = requests.get(
                    "https://nominatim.openstreetmap.org/search",
                    params=params,
                    headers=headers,
                    timeout=8,
                )
                if resp.json():
                    nominatim_found = True
            except:
                pass

        combined_confidence = 0.85 if scraper else (0.60 if nominatim_found else 0.10)
        groq_val = bool(groq_result.get("has_google_maps")) if groq_result else None
        if groq_val:
            combined_confidence = min(1.0, combined_confidence + 0.20)
        passed = combined_confidence >= 0.40
        sources = (
            ["scraper"]
            + (["nominatim"] if nominatim_found else [])
            + (["groq"] if groq_val is not None else [])
        )
        result = ConfidenceResult(passed, combined_confidence, sources)
        _log_check("Google My Business listing", result)
        return result

    def analyze(self):
        if not self.crawl():
            return {
                "score": 0,
                "seo_score": 0,
                "engagement_score": 0,
                "criteria": [
                    {
                        "text": "Unable to access website",
                        "passed": False,
                        "points": 0,
                        "confidence": 0.0,
                        "sources": [],
                    }
                ],
                "error": "Could not access website",
            }

        combined_text = ""
        for url, html in self.pages_content.items():
            page_text = BeautifulSoup(html, "html.parser").get_text(
                separator=" ", strip=True
            )
            combined_text += f"\n\n--- PAGE: {url} ---\n{page_text[:4000]}\n"

        groq_result = None
        analysis_method = "hybrid-rule-based"
        ai_summary = None
        ai_suggestions = None

        if check_groq_available():
            groq_result = analyze_with_groq(combined_text, self.base_url, self.software)
            if groq_result:
                analysis_method = "hybrid-groq"
                ai_summary = groq_result.get("summary", "")
                ai_suggestions = json.dumps(
                    groq_result.get("improvement_suggestions", [])
                )

        criteria_map = [
            ("Mobile-responsive website", self.check_mobile_responsive, 10),
            ("Online appointment booking", self.check_online_booking, 15),
            ("Digital patient intake (online)", self.check_digital_intake, 15),
            ("Patient forms available", self.check_patient_forms, 5),
            ("Modern check-in system", self.check_checkin, 10),
            ("Active social media presence", self.check_social_media, 10),
            ("Patient testimonials/reviews", self.check_testimonials, 10),
            ("Educational content/blog", self.check_blog, 10),
            ("Video content about services", self.check_video, 5),
            ("SSL certificate (HTTPS)", self.check_ssl, 5),
            ("Contact information clearly visible", self.check_contact_info, 3),
            ("Google My Business listing", self.check_google_maps, 2),
        ]

        score = 0
        criteria = []
        for label, check_fn, points in criteria_map:
            try:
                cr = check_fn(groq_result=groq_result)
                if cr.passed:
                    score += points
                criteria.append(
                    {
                        "text": label,
                        "passed": cr.passed,
                        "points": points,
                        "confidence": cr.confidence,
                        "sources": cr.sources,
                    }
                )
            except Exception as e:
                criteria.append(
                    {
                        "text": label,
                        "passed": False,
                        "points": points,
                        "confidence": 0.0,
                        "sources": [],
                    }
                )

        SEO_WEIGHTS = {
            "Mobile-responsive website": 15,
            "SSL certificate (HTTPS)": 15,
            "Google My Business listing": 20,
            "Educational content/blog": 15,
            "Patient testimonials/reviews": 15,
            "Active social media presence": 10,
            "Contact information clearly visible": 10,
        }
        seo_raw = sum(
            SEO_WEIGHTS.get(c["text"], 0)
            for c in criteria
            if c["passed"] and c["text"] in SEO_WEIGHTS
        )
        seo_score = round((seo_raw / sum(SEO_WEIGHTS.values())) * 100)

        ENG_WEIGHTS = {
            "Online appointment booking": 30,
            "Digital patient intake (online)": 25,
            "Modern check-in system": 20,
            "Video content about services": 15,
            "Patient testimonials/reviews": 10,
        }
        eng_raw = sum(
            ENG_WEIGHTS.get(c["text"], 0)
            for c in criteria
            if c["passed"] and c["text"] in ENG_WEIGHTS
        )
        engagement_score = round((eng_raw / sum(ENG_WEIGHTS.values())) * 100)

        print(f"\n🎯 Score:{score} SEO:{seo_score} Eng:{engagement_score}", flush=True)
        pagespeed_data = get_google_pagespeed_scores(self.base_url)
        # After calculating score from criteria:
        mobile_ps  = pagespeed_data.get('mobile',  {}).get('score', 0)
        desktop_ps = pagespeed_data.get('desktop', {}).get('score', 0)

        # Penalize score if PageSpeed is bad
        if mobile_ps > 0 and mobile_ps < 50:
            score = max(0, score - 15)   # heavy penalty
        elif mobile_ps < 70:
            score = max(0, score - 8)    # moderate penalty

        if desktop_ps > 0 and desktop_ps < 50:
            score = max(0, score - 10)
        elif desktop_ps < 70:
            score = max(0, score - 5)

        return {
            "score": score,
            "seo_score": seo_score,
            "engagement_score": engagement_score,
            "criteria": criteria,
            "url": self.base_url,
            "pages_analyzed": len(self.visited_urls),
            "pagespeed": pagespeed_data,
            "ai_summary": ai_summary,
            "ai_suggestions": json.loads(ai_suggestions) if ai_suggestions else [],
            "analysis_method": analysis_method,
            "software": self.software,
        }


def _log_check(label: str, result: ConfidenceResult):
    icon = "✓" if result.passed else "✗"
    print(
        f"  {icon} {label:<40} conf={result.confidence:.0%}  [{', '.join(result.sources)}]",
        flush=True,
    )


# ==================== BACKGROUND THREAD ====================
def run_analysis_in_background(analysis_id, url, form_data):
    try:
        print(f"\n🚀 Starting background analysis for {url}", flush=True)
        analysis_status[analysis_id] = {"status": "analyzing", "progress": 0}
        practice_name = form_data.get("practiceName", "")
        software = form_data.get("currentSoftware", "")
        analyzer = MultiPageAnalyzer(
            url, max_pages=8, practice_name=practice_name, software=software
        )
        results = analyzer.analyze()

        with app.app_context():
            lead = Lead(
                practice_name=practice_name,
                website_url=url,
                provider_name=form_data.get("providerName", ""),
                email=form_data.get("email", ""),
                phone=form_data.get("phone", ""),
                zip_code=form_data.get("zip", ""),
                current_software=software,
                score=results["score"],
                seo_score=results.get("seo_score"),
                engagement_score=results.get("engagement_score"),
                ai_summary=results.get("ai_summary"),
                ai_suggestions=json.dumps(results.get("ai_suggestions", [])),
                analysis_method=results.get("analysis_method", "hybrid-rule-based"),
            )
            db.session.add(lead)
            db.session.commit()
            for criterion in results["criteria"]:
                db.session.add(
                    AnalysisResult(
                        lead_id=lead.id,
                        criterion_name=criterion["text"],
                        passed=criterion["passed"],
                        confidence=criterion.get("confidence", 0.0),
                        sources=",".join(criterion.get("sources", [])),
                        points=criterion["points"],
                    )
                )
            db.session.commit()

        analysis_status[analysis_id] = {"status": "complete", "results": results}
        print(f"✅ Analysis complete for {url}", flush=True)
    except Exception as e:
        print(f"❌ Error: {e}", flush=True)
        import traceback

        traceback.print_exc()
        analysis_status[analysis_id] = {"status": "error", "error": str(e)}


# ==================== ROUTES ====================
@app.route("/")
def index():
    try:
        return send_from_directory(".", "frontend/index.html")
    except:
        return "HTML file not found", 404


@app.route("/admin")
def admin():
    try:
        return send_from_directory(".", "frontend/admin_dashboard.html")
    except:
        return "Admin dashboard not found", 404


@app.route("/<path:path>")
def serve_static(path):
    try:
        return send_from_directory(".", path)
    except:
        return "File not found", 404


# ==================== OTP ENDPOINTS ====================
@app.route("/api/request-otp", methods=["POST", "OPTIONS"])
def request_otp():
    if request.method == "OPTIONS":
        return "", 200
    try:
        data = request.json
        phone = data.get("phone", "").strip()
        if not phone:
            return jsonify({"success": False, "message": "Phone number required"}), 400
        cleanup_expired_otps()
        otp_code = generate_otp(6)
        otp_storage[phone] = {
            "code": otp_code,
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(minutes=10),
            "attempts": 0,
        }
        print(
            f"\n{'='*60}\n🧪 TEST MODE - OTP: {otp_code}  Phone: {phone}\n{'='*60}\n",
            flush=True,
        )
        return jsonify(
            {"success": True, "message": "OTP generated (check console for code)"}
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


@app.route("/api/verify-otp", methods=["POST", "OPTIONS"])
def verify_otp():
    if request.method == "OPTIONS":
        return "", 200
    try:
        data = request.json
        phone = data.get("phone", "").strip()
        otp = data.get("otp", "").strip()
        if not phone or not otp:
            return jsonify({"success": False, "message": "Phone and OTP required"}), 400
        if len(otp) != 6 or not otp.isdigit():
            return jsonify({"success": False, "message": "Invalid OTP format"}), 400
        if phone not in otp_storage:
            return (
                jsonify(
                    {"success": False, "message": "No OTP found. Request a new one."}
                ),
                400,
            )
        otp_data = otp_storage[phone]
        if datetime.utcnow() > otp_data["expires_at"]:
            del otp_storage[phone]
            return jsonify({"success": False, "message": "OTP expired."}), 400
        if otp_data["attempts"] >= 3:
            del otp_storage[phone]
            return jsonify({"success": False, "message": "Too many attempts."}), 400
        if otp_data["code"] == otp:
            del otp_storage[phone]
            return jsonify({"success": True, "message": "Phone verified successfully"})
        else:
            otp_data["attempts"] += 1
            remaining = 3 - otp_data["attempts"]
            if remaining > 0:
                return (
                    jsonify(
                        {
                            "success": False,
                            "message": f"Invalid OTP. {remaining} attempt(s) remaining.",
                        }
                    ),
                    400,
                )
            else:
                del otp_storage[phone]
                return (
                    jsonify({"success": False, "message": "Too many failed attempts."}),
                    400,
                )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ==================== ANALYSIS ENDPOINTS ====================
@app.route("/api/analyze-website", methods=["POST", "OPTIONS"])
def analyze_website():
    if request.method == "OPTIONS":
        return "", 200
    try:
        data = request.json
        website_url = data.get("websiteUrl", "").strip()
        if not website_url:
            return jsonify({"error": "Website URL required"}), 400
        if not website_url.startswith(("http://", "https://")):
            website_url = "https://" + website_url
        analysis_id = f"{int(time.time())}_{hash(website_url)}"
        analysis_status[analysis_id] = {"status": "analyzing", "progress": 0}
        thread = threading.Thread(
            target=run_analysis_in_background, args=(analysis_id, website_url, data)
        )
        thread.daemon = True
        thread.start()
        return jsonify(
            {
                "analysis_id": analysis_id,
                "status": "started",
                "message": "Analysis started",
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analysis-status/<analysis_id>", methods=["GET"])
def get_analysis_status(analysis_id):
    if analysis_id not in analysis_status:
        return jsonify({"error": "Analysis not found"}), 404
    return jsonify(analysis_status[analysis_id])


@app.route("/api/leads", methods=["GET"])
def get_leads():
    try:
        page = request.args.get("page", 1, type=int)
        per_page = request.args.get("per_page", 50, type=int)
        leads = Lead.query.order_by(Lead.created_at.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )
        return jsonify(
            {
                "leads": [l.to_dict() for l in leads.items],
                "total": leads.total,
                "pages": leads.pages,
                "current": leads.page,
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/leads/<int:lead_id>", methods=["GET"])
def get_lead(lead_id):
    try:
        lead = Lead.query.get_or_404(lead_id)
        results = AnalysisResult.query.filter_by(lead_id=lead_id).all()
        return jsonify(
            {
                "lead": lead.to_dict(),
                "analysis": [
                    {
                        "criterion_name": r.criterion_name,
                        "passed": r.passed,
                        "confidence": r.confidence,
                        "sources": r.sources.split(",") if r.sources else [],
                        "points": r.points,
                    }
                    for r in results
                ],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/leads/<int:lead_id>", methods=["DELETE"])
def delete_lead(lead_id):
    try:
        lead = Lead.query.get_or_404(lead_id)
        db.session.delete(lead)
        db.session.commit()
        return jsonify({"success": True, "message": f"Lead {lead_id} deleted"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@app.route("/api/leads/<int:lead_id>/corrections", methods=["GET"])
def get_corrections(lead_id):
    try:
        return jsonify(
            [c.to_dict() for c in CorrectionLog.query.filter_by(lead_id=lead_id).all()]
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/leads/<int:lead_id>/corrections", methods=["POST"])
def save_corrections(lead_id):
    try:
        data = request.json
        corrected_by = data.get("corrected_by", "admin")
        corrections = data.get("corrections", [])
        if not corrections:
            return jsonify({"error": "No corrections provided"}), 400
        Lead.query.get_or_404(lead_id)
        saved = []
        for corr in corrections:
            criterion_name = corr.get("criterion_name")
            human_decision = corr.get("human_decision")
            if criterion_name is None or human_decision is None:
                continue
            ar = AnalysisResult.query.filter_by(
                lead_id=lead_id, criterion_name=criterion_name
            ).first()
            ai_decision = ar.passed if ar else None
            ai_confidence = ar.confidence if ar else 0.0
            if ai_decision is not None and ai_decision == human_decision:
                continue
            existing = CorrectionLog.query.filter_by(
                lead_id=lead_id, criterion_name=criterion_name
            ).first()
            if existing:
                existing.human_decision = human_decision
                existing.reason = corr.get("reason", existing.reason)
                existing.corrected_by = corrected_by
                existing.created_at = datetime.utcnow()
                saved.append(existing.to_dict())
            else:
                new_corr = CorrectionLog(
                    lead_id=lead_id,
                    criterion_name=criterion_name,
                    ai_decision=ai_decision if ai_decision is not None else False,
                    human_decision=human_decision,
                    ai_confidence=ai_confidence,
                    reason=corr.get("reason", ""),
                    corrected_by=corrected_by,
                )
                db.session.add(new_corr)
                saved.append({"criterion_name": criterion_name, "saved": True})
        db.session.commit()
        return jsonify({"success": True, "saved": len(saved), "corrections": saved})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


# ==================== RANK CHECK ENDPOINT ====================
@app.route("/api/rank-check", methods=["POST", "OPTIONS"])
def rank_check():
    if request.method == "OPTIONS":
        return "", 200
    try:
        data = request.json or {}
        domain = (
            data.get("domain", "")
            .strip()
            .lower()
            .replace("https://", "")
            .replace("http://", "")
            .split("/")[0]
        )
        keywords = data.get("keywords", [])
        location = data.get("location", "India")
        device = data.get("device", "mobile")
        lead_id = data.get('lead_id')
        if not lead_id:
            # Auto-match lead by domain
            try:
                matched_lead = Lead.query.filter(
                    Lead.website_url.ilike(f'%{domain}%')
                ).order_by(Lead.created_at.desc()).first()
                if matched_lead:
                    lead_id = matched_lead.id
                    log.info(f"  🔗 Auto-matched lead_id={lead_id} for domain={domain}")
                else:
                    log.info(f"  ℹ️  No lead found for domain={domain} — lead_id stays NULL")
            except Exception as e:
                log.warning(f"  ⚠️  Lead lookup failed: {e}")

        if not domain:
            return jsonify({"error": "domain is required"}), 400
        if not keywords:
            return jsonify({"error": "at least one keyword required"}), 400
        if len(keywords) > 10:
            return jsonify({"error": "max 10 keywords"}), 400

        print(
            f"\\n🔍 SerpApi Rank check: {domain} | {len(keywords)} kw | {location} | {device}",
            flush=True,
        )

        results = []
        for kw in keywords:
            kw = kw.strip()
            if not kw:
                continue

            # ── SerpApi → fallback ──
            r = get_rank_via_serpapi(kw, domain, location, device)

            # Pretty log
            organic_str = f"#{r['rank']}" if r.get("rank") is not None else "—"
            local_str = f" [local #{r['local_rank']}]" if r.get("local_rank") else ""
            feat_str = " ⭐FEATURED" if r.get("featured") else ""
            print(
                f"  {'✓' if r['found'] else '✗'} [{organic_str:>5}]{local_str}{feat_str}  {kw}",
                flush=True,
            )

            results.append(r)

        metrics = calculate_visibility_score(results)

        # ── Save to MySQL ──
        try:
            rc = RankCheck(
                lead_id=lead_id,
                domain=domain,
                location=location,
                device=device,
                keywords_json=json.dumps(keywords),
                results_json=json.dumps(results),
                metrics_json=json.dumps(metrics),
                visibility_score=metrics["visibility_score"],
                avg_rank=metrics["avg_rank"],
                live_data=bool(SERPAPI_KEY),
            )
            db.session.add(rc)
            db.session.commit()
            print(f"  💾 Saved rank check (id={rc.id})", flush=True)
        except Exception as db_err:
            print(f"  ⚠️  DB save failed: {db_err}", flush=True)
            db.session.rollback()

        return jsonify(
            {
                "domain": domain,
                "location": location,
                "device": device,
                "results": results,
                "metrics": metrics,
                "live_data": bool(SERPAPI_KEY),
                "engine": "serpapi" if SERPAPI_KEY else "estimated",
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/rank-history/<domain>", methods=["GET"])
def rank_history(domain):
    """Get last 10 rank checks for a domain."""
    try:
        checks = (
            RankCheck.query.filter_by(domain=domain)
            .order_by(RankCheck.created_at.desc())
            .limit(10)
            .all()
        )
        return jsonify([c.to_dict() for c in checks])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==================== CHATBOT ENDPOINT ====================
@app.route("/api/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return "", 200
    try:
        data = request.json
        messages = data.get("messages", [])
        if not messages:
            return jsonify({"error": "No messages provided"}), 400

        client = Groq(api_key=GROQ_API_KEY)
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """You are ZoΞ, an expert AI assistant embedded in the RankPulse website — a platform that analyzes healthcare practice websites and gives them a score out of 100. You are created by Ayush Panwar to know more about him connect him on LinkedIn: https://www.linkedin.com/in/ayush-panwar603957264/

IMPORTANT RULE: If the user asks about THEIR specific scores, results, or analysis — do NOT make up numbers. Tell them to fill the diagnostic form first.

Use this response when they ask about their own scores:
"To see your actual scores, you'll need to run a free analysis first! 🏥
👉 Click 'Diagnose Your Practice' in the top nav and fill in your practice details — it only takes 2 minutes.
Once done, I can help you interpret your real results!"

You help practice owners understand:
- How each score is calculated and what affects it
- How to improve their online presence
- Benefits of features like online booking, digital intake, etc.
- General industry benchmarks (NOT made-up specific numbers)

Score system:
- Overall score (0-100): 12 criteria checks
- SEO score: mobile-responsive (15pts), SSL (15pts), GMB (20pts), blog (15pts), testimonials (15pts), social media (10pts), contact info (10pts)
- PageSpeed: fetched live from Google PageSpeed Insights API
- Engagement: online booking (30pts), digital intake (25pts), check-in (20pts), video (15pts), testimonials (10pts)

Keep responses concise, practical, and warm.""",
                }
            ]
            + messages[-10:],
            temperature=0.7,
            max_tokens=600,
        )
        reply = response.choices[0].message.content
        return jsonify({"reply": reply})
    except Exception as e:
        print(f"Chat error: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


# ==================== EXPORT + STATS ====================
@app.route("/api/export-dataset", methods=["GET"])
def export_dataset():
    try:
        fmt = request.args.get("format", "jsonl")
        corrections = CorrectionLog.query.order_by(CorrectionLog.created_at).all()
        examples = []
        for c in corrections:
            lead = Lead.query.get(c.lead_id)
            if not lead:
                continue
            prompt = f"Analyze whether the following healthcare practice has '{c.criterion_name}'.\n\nPractice: {lead.practice_name}\nWebsite: {lead.website_url}\nSoftware: {lead.current_software or 'unknown'}\nAI initially said: {'YES' if c.ai_decision else 'NO'} (confidence: {c.ai_confidence:.0%})\n"
            completion = f"Correct answer: {'YES' if c.human_decision else 'NO'}.\nReason: {c.reason or 'No reason provided.'}"
            examples.append(
                {
                    "prompt": prompt,
                    "completion": completion,
                    "metadata": {
                        "lead_id": c.lead_id,
                        "criterion": c.criterion_name,
                        "ai_was_wrong": c.ai_decision != c.human_decision,
                        "ai_confidence": c.ai_confidence,
                        "corrected_by": c.corrected_by,
                        "corrected_at": c.created_at.isoformat(),
                    },
                }
            )
        if fmt == "jsonl":
            lines = "\n".join(json.dumps(ex) for ex in examples)
            return Response(
                lines,
                mimetype="application/json",
                headers={
                    "Content-Disposition": "attachment; filename=corrections_dataset.jsonl"
                },
            )
        return jsonify({"total_examples": len(examples), "examples": examples})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/correction-stats", methods=["GET"])
def correction_stats():
    try:
        corrections = CorrectionLog.query.all()
        stats = {}
        for c in corrections:
            name = c.criterion_name
            if name not in stats:
                stats[name] = {
                    "criterion": name,
                    "total_reviewed": 0,
                    "ai_wrong": 0,
                    "ai_false_neg": 0,
                    "ai_false_pos": 0,
                    "avg_confidence": [],
                }
            stats[name]["total_reviewed"] += 1
            if c.ai_decision != c.human_decision:
                stats[name]["ai_wrong"] += 1
                if not c.ai_decision and c.human_decision:
                    stats[name]["ai_false_neg"] += 1
                else:
                    stats[name]["ai_false_pos"] += 1
            stats[name]["avg_confidence"].append(c.ai_confidence)
        for name in stats:
            confs = stats[name]["avg_confidence"]
            stats[name]["avg_confidence"] = (
                round(sum(confs) / len(confs), 2) if confs else 0
            )
            total = stats[name]["total_reviewed"]
            wrong = stats[name]["ai_wrong"]
            stats[name]["accuracy"] = (
                round((total - wrong) / total, 2) if total else 1.0
            )
        return jsonify(
            {
                "total_corrections": len(corrections),
                "criteria_stats": sorted(
                    stats.values(), key=lambda x: x["ai_wrong"], reverse=True
                ),
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/software-capabilities", methods=["GET"])
def get_software_capabilities():
    serialisable = {}
    for sw, caps in SOFTWARE_CAPABILITIES.items():
        serialisable[sw] = {
            k: ("partial" if isinstance(v, _Partial) else v) for k, v in caps.items()
        }
    return jsonify(serialisable)


@app.route("/api/health", methods=["GET"])
def health_check():
    # Test MySQL connection
    db_status = "connected"
    try:
        db.session.execute(db.text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {str(e)}"

    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "database": "mysql",
            "db_status": db_status,
            "groq_configured": check_groq_available(),
            "groq_model": GROQ_MODEL,
            "google_cse": bool(GOOGLE_API_KEY and GOOGLE_CX),
            "software_count": len(SOFTWARE_CAPABILITIES),
        }
    )


# =====================Rank Endpoints====================
_predictor = None  # global model instance
_predictor_lock = threading.Lock()


def get_predictor():
    """Lazy-load the predictor. Returns None if not trained yet."""
    global _predictor
    model_path = Path("./models")
    if _predictor is None and (model_path / "classifier.pkl").exists():
        with _predictor_lock:
            if _predictor is None:
                _predictor = RankPredictor.load(model_path)
    return _predictor


@app.route("/api/ml/train", methods=["POST"])
def ml_train():
    """
    Train / retrain the rank prediction model.
    POST body (optional): { "use_synthetic": true, "model_type": "xgb" }
    """
    data = request.json or {}
    use_synthetic = data.get("use_synthetic", False)
    model_type = data.get("model_type", "xgb")

    def _train_thread():
        global _predictor
        try:
            # Load real data
            df_raw = load_from_mysql()
            df_exploded = (
                explode_results(df_raw) if not df_raw.empty else pd.DataFrame()
            )

            # Merge with synthetic if not enough real data
            if len(df_exploded) < 200 or use_synthetic:
                n_synthetic = max(0, 2000 - len(df_exploded))
                df_synth = generate_synthetic_data(n_synthetic + 1000)
                df_train = (
                    pd.concat([df_exploded, df_synth], ignore_index=True)
                    if not df_exploded.empty
                    else df_synth
                )
                log.info(
                    f"  Using {len(df_exploded)} real + {len(df_synth)} synthetic rows"
                )
            else:
                df_train = df_exploded
                log.info(f"  Using {len(df_train)} real rows only")

            predictor = RankPredictor(model_type=model_type)
            meta = predictor.train(df_train)
            predictor.save()
            with _predictor_lock:
                _predictor = predictor
            log.info("✅ ML training complete")
        except Exception as e:
            log.error(f"ML training error: {e}")
            import traceback

            traceback.print_exc()

    thread = threading.Thread(target=_train_thread)
    thread.daemon = True
    thread.start()
    return jsonify(
        {
            "status": "training_started",
            "message": "Training in background — check /api/ml/status",
        }
    )


@app.route("/api/ml/status", methods=["GET"])
def ml_status():
    """Check if model is trained and return metadata."""
    p = get_predictor()
    if p is None:
        return jsonify(
            {
                "trained": False,
                "message": "No model found. POST to /api/ml/train first.",
            }
        )
    return jsonify({"trained": True, **p.training_meta})


@app.route("/api/ml/predict", methods=["POST"])
def ml_predict():
    """
    Predict rank for a (domain, keyword) pair.

    POST body:
    {
        "keyword":        "chiropractor near me",
        "has_ssl":        true,
        "has_mobile":     true,
        "has_gmb":        true,
        "has_blog":       false,
        "has_testimonials": true,
        "has_booking":    true,
        "has_intake":     false,
        "has_video":      false,
        "has_social":     true,
        "has_contact":    true,
        "pagespeed_mobile":  55,
        "pagespeed_desktop": 72,
        "website_score":  65,
        "seo_score":      58,
        "engagement_score": 45,
        "device":         "mobile",
        "location":       "Austin TX"
    }
    """
    p = get_predictor()
    if p is None:
        return (
            jsonify({"error": "Model not trained. POST to /api/ml/train first."}),
            503,
        )

    features = request.json or {}
    if not features.get("keyword"):
        return jsonify({"error": "keyword is required"}), 400

    try:
        result = p.predict(features)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ml/predict-bulk", methods=["POST"])
def ml_predict_bulk():
    """
    Predict ranks for multiple keywords for a domain.

    POST body:
    {
        "base_features": { ...website signals... },
        "keywords": ["chiropractor near me", "back pain specialist", "spine doctor"]
    }
    """
    p = get_predictor()
    if p is None:
        return jsonify({"error": "Model not trained"}), 503

    data = request.json or {}
    base = data.get("base_features", {})
    keywords = data.get("keywords", [])

    if not keywords:
        return jsonify({"error": "keywords array required"}), 400

    results = []
    for kw in keywords[:20]:  # cap at 20
        features = {**base, "keyword": kw}
        try:
            pred = p.predict(features)
            results.append({"keyword": kw, **pred})
        except Exception as e:
            results.append({"keyword": kw, "error": str(e)})

    return jsonify({"predictions": results, "count": len(results)})


@app.route("/api/ml/feature-importance", methods=["GET"])
def ml_feature_importance():
    """Return top feature importances from the trained classifier."""
    p = get_predictor()
    if p is None:
        return jsonify({"error": "Model not trained"}), 503
    return jsonify({"importances": p.feature_importance()})


@app.route("/api/ml/retrain-schedule", methods=["POST"])
def ml_retrain_schedule():
    """Trigger weekly auto-retrain. Call from a cron job or scheduler."""
    # Check if model was trained > 7 days ago
    p = get_predictor()
    if p and p.training_meta.get("trained_at"):
        trained_at = datetime.fromisoformat(p.training_meta["trained_at"])
        age_days = (datetime.now() - trained_at).days
        if age_days < 7:
            return jsonify(
                {
                    "skipped": True,
                    "reason": f"Model is only {age_days} days old",
                    "trained_at": p.training_meta["trained_at"],
                }
            )

    # Trigger retrain
    from flask import current_app

    with current_app.test_request_context():
        return ml_train()


# ==================== COMPETITOR ENDPOINTS ====================
SPECIALTY_TAGS = {
    "chiro": [
        '"healthcare"="yes"',
        '"amenity"="doctors"',
        '"healthcare:speciality"="chiropractic"',
    ],
    "physio": [
        '"healthcare:speciality"="physiotherapy"',
        '"amenity"="physiotherapist"',
    ],
    "dental": ['"amenity"="dentist"'],
    "optometry": ['"amenity"="optometrist"'],
    "general": ['"amenity"="doctors"', '"amenity"="clinic"', '"healthcare"="yes"'],
}


def get_coords_from_zip(zip_code):
    try:
        params = {
            "q": f"{zip_code}, USA",
            "format": "json",
            "limit": 1,
            "addressdetails": 0,
        }
        headers = {"User-Agent": "RankPulse -Competitor-Finder/1.0"}
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params=params,
            headers=headers,
            timeout=10,
        )
        data = resp.json()
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
    except Exception as e:
        print(f"  ⚠️  Nominatim error: {e}", flush=True)
    return None, None


def overpass_nearby_practices(lat, lng, radius_m=5000, specialty="general"):
    tags = SPECIALTY_TAGS.get(specialty, SPECIALTY_TAGS["general"])
    tag_filters = "\n  ".join(
        f"node[{t}](around:{radius_m},{lat},{lng});\n  way[{t}](around:{radius_m},{lat},{lng});"
        for t in tags
    )
    query = f"\n[out:json][timeout:25];\n(\n  {tag_filters}\n);\nout center tags;\n"

    max_retries = 3
    for attempt in range(max_retries):
        try:
            wait = 2 ** attempt  # 1s, 2s, 4s
            if attempt > 0:
                print(f"  ⏳ Overpass retry {attempt}/{max_retries - 1} (wait {wait}s)...", flush=True)
                time.sleep(wait)

            resp = requests.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": query},
                timeout=30,
                headers={"User-Agent": "RankPulse -Competitor-Finder/1.0"},
            )

            # Check HTTP status first
            if resp.status_code == 429:
                print(f"  ⚠️  Overpass rate-limited (429) — backing off", flush=True)
                continue
            resp.raise_for_status()

            # Guard against empty body (Overpass occasionally returns blank on overload)
            raw = resp.text.strip()
            if not raw:
                print(f"  ⚠️  Overpass returned empty body on attempt {attempt + 1}", flush=True)
                continue

            data = resp.json()

            elements = data.get("elements", [])
            results = []
            for el in elements:
                tags_data = el.get("tags", {})
                name = tags_data.get("name", "").strip()
                website = (
                    tags_data.get("website") or tags_data.get("contact:website") or ""
                ).strip()
                if not name:
                    continue
                el_lat = (
                    el.get("center", {}).get("lat", lat)
                    if el.get("type") == "way"
                    else el.get("lat", lat)
                )
                el_lng = (
                    el.get("center", {}).get("lon", lng)
                    if el.get("type") == "way"
                    else el.get("lon", lng)
                )
                dist = haversine_m(lat, lng, el_lat, el_lng)
                results.append({
                    "name": name,
                    "website": website,
                    "lat": el_lat,
                    "lng": el_lng,
                    "distance_m": round(dist),
                    "tags": tags_data,
                })

            results.sort(key=lambda x: x["distance_m"])
            seen = set()
            unique = []
            for r in results:
                key = r["name"].lower().strip()
                if key not in seen:
                    seen.add(key)
                    unique.append(r)
            return unique

        except requests.exceptions.Timeout:
            print(f"  ⚠️  Overpass timeout on attempt {attempt + 1}", flush=True)
        except requests.exceptions.ConnectionError as e:
            print(f"  ⚠️  Overpass connection error: {e}", flush=True)
        except ValueError as e:
            # JSON parse failure — log the raw response for debugging
            raw_preview = resp.text[:200] if resp else "(no response)"
            print(f"  ⚠️  Overpass JSON parse error: {e} | raw: {raw_preview!r}", flush=True)
        except Exception as e:
            print(f"  ⚠️  Overpass error: {e}", flush=True)

    print(f"  ✗ Overpass failed after {max_retries} attempts — using fallback", flush=True)
    return []

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def detect_specialty(software_name, website_url=""):
    combined = (software_name + " " + website_url).lower()
    if any(k in combined for k in ["chiro", "chirotouch", "chirospring", "genesis"]):
        return "chiro"
    if any(k in combined for k in ["physio", "rehab", "pt "]):
        return "physio"
    if any(k in combined for k in ["dental", "dentist", "ortho"]):
        return "dental"
    return "general"


def score_competitor(competitor, practice_url):
    website = competitor.get("website", "").strip()
    if website:
        if not website.startswith(("http://", "https://")):
            website = "https://" + website
        try:
            analyzer = MultiPageAnalyzer(website, max_pages=3)
            results = analyzer.analyze()
            return {"score": results["score"], "website": website, "method": "scraped"}
        except:
            pass
    dist_km = competitor.get("distance_m", 5000) / 1000
    est_score = max(30, min(75, int(70 - dist_km * 1.5)))
    return {"score": est_score, "website": website or "N/A", "method": "estimated"}


def _badge(score, my_score):
    if score > my_score + 10:
        return "Industry Leader"
    if score > my_score + 5:
        return "Above Average"
    if score >= my_score - 5:
        return "Similar Level"
    return "Below Average"


@app.route("/api/competitors", methods=["POST", "OPTIONS"])
def find_competitors():
    if request.method == "OPTIONS":
        return "", 200
    try:
        data = request.json or {}
        zip_code = data.get("zip", "").strip()
        software = data.get("currentSoftware", "")
        my_url = data.get("websiteUrl", "")
        my_score = int(data.get("myScore", 0))
        radius = int(data.get("radius", 5000))
        max_comps = int(data.get("maxCompetitors", 3))
        if not zip_code:
            return jsonify({"error": "zip is required"}), 400

        lat, lng = get_coords_from_zip(zip_code)
        if lat is None:
            return jsonify({"error": f"Could not geocode ZIP {zip_code}"}), 400

        specialty = detect_specialty(software, my_url)
        my_domain = (
            urlparse(my_url).netloc.replace("www.", "").lower() if my_url else ""
        )

        nearby = []
        for search_radius, search_specialty in [
            (radius, specialty),
            (radius * 2, specialty),
            (radius * 2, "general"),
        ]:
            nearby = overpass_nearby_practices(
                lat, lng, radius_m=search_radius, specialty=search_specialty
            )
            nearby = [
                p for p in nearby if my_domain not in p.get("website", "").lower()
            ]
            if nearby:
                break

        competitors = []
        for idx, comp in enumerate(nearby[: min(max_comps, 5)]):
            scored = score_competitor(comp, my_url)
            competitors.append(
                {
                    "rank": idx + 1,
                    "name": comp["name"],
                    "score": scored["score"],
                    "distance_m": comp["distance_m"],
                    "website": scored["website"],
                    "method": scored["method"],
                    "badge": _badge(scored["score"], my_score),
                }
            )

        if not competitors:
            import random

            random.seed(42)
            offsets = sorted(
                [random.randint(-15, 20) for _ in range(max_comps)], reverse=True
            )
            competitors = [
                {
                    "rank": i + 1,
                    "name": f"Nearby Practice {i+1}",
                    "score": max(10, min(100, 65 + offsets[i])),
                    "distance_m": (i + 1) * 800,
                    "website": "N/A",
                    "method": "estimated",
                    "badge": "",
                }
                for i in range(max_comps)
            ]

        competitors.sort(key=lambda x: x["score"], reverse=True)
        for i, c in enumerate(competitors):
            c["rank"] = i + 1
            c["badge"] = _badge(c["score"], my_score)
        all_scores = [c["score"] for c in competitors] + [my_score]
        industry_avg = round(sum(all_scores) / len(all_scores))

        return jsonify(
            {
                "my_score": my_score,
                "competitors": competitors[:max_comps],
                "industry_avg": industry_avg,
                "specialty": specialty,
                "coords": {"lat": lat, "lng": lng},
                "radius_m": radius,
            }
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ==================== INIT ====================
with app.app_context():
    try:
        db.create_all()
        db.session.execute(db.text("SELECT 1"))
        print(f"✓ Database ready ({'MySQL' if DB_AVAILABLE else 'SQLite fallback'})", flush=True)
    except Exception as e:
        print(f"✗ Database init FAILED: {e}", flush=True)


if __name__ == "__main__":
    print(f"\n{'='*60}\n  RankPulse  — MySQL Edition\n{'='*60}", flush=True)
    print(f"  DB: mysql://{DB_HOST}:{DB_PORT}/{DB_NAME}", flush=True)
    print(f"  Public Form:     http://localhost:5000/", flush=True)
    print(f"  Admin Dashboard: http://localhost:5000/admin", flush=True)
    print(f"\n  Press CTRL+C to stop\n{'='*60}\n", flush=True)
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
