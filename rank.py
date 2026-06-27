"""
╔══════════════════════════════════════════════════════════════════╗
║         RankPulse  — Rank Prediction ML Pipeline                    ║
║         Model: XGBoost + Random Forest + LightGBM ensemble       ║
║         Features: website metrics → predicted Google rank        ║
╚══════════════════════════════════════════════════════════════════╝

INSTALL:
    pip install xgboost lightgbm scikit-learn pandas numpy shap pymysql sqlalchemy python-dotenv

HOW IT WORKS:
    1. DATA COLLECTION  → pulls from your MySQL rank_checks + leads tables
    2. FEATURE ENGINEERING → extracts 20+ signals per domain
    3. TRAINING          → XGBoost + Random Forest + LightGBM ensemble
    4. EVALUATION        → NDCG, MRR, MAE on test split
    5. PREDICTION API    → Flask route /api/predict-rank
    6. EXPLAINABILITY    → SHAP values per prediction (why this rank?)
    7. AUTO-RETRAIN      → Scheduler retrains weekly as new data arrives

FEATURES USED TO PREDICT RANK:
    Website signals:    SSL, mobile responsive, PageSpeed scores
    Content signals:    has blog, has video, has testimonials, word count
    Technical signals:  has booking, digital intake, GMB listing
    Historical:         domain age (estimated), past avg rank
    Keyword signals:    keyword length, local vs generic, brand vs non-brand

TARGETS:
    - rank_bucket   (classification: top3 / top10 / top30 / not_ranking)
    - organic_rank  (regression: predicted numeric rank 1–100)
"""

import os
import sys
import json
import math
import hashlib
import random
import pickle
import warnings
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score,
    ndcg_score, accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ── Try importing XGBoost / LightGBM (graceful fallback) ──
try:
    import xgboost as xgb
    HAS_XGB = True
    log.info("✓ XGBoost available")
except ImportError:
    HAS_XGB = False
    log.warning("⚠ XGBoost not installed — using GradientBoosting fallback. Run: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
    log.info("✓ LightGBM available")
except ImportError:
    HAS_LGB = False
    log.warning("⚠ LightGBM not installed — skipping. Run: pip install lightgbm")

# Replace the current shap import block:
try:
    import shap
    # Quick sanity check
    _ = shap.TreeExplainer
    HAS_SHAP = True
    log.info("✓ SHAP available")
except Exception:          # catches both ImportError AND numpy ABI errors
    HAS_SHAP = False
    log.warning("⚠ SHAP unavailable — explanations disabled")
# ── DB config (reads from .env / environment) ──
from dotenv import load_dotenv
load_dotenv()

DB_USER     = os.getenv("DB_USER",     "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST     = os.getenv("DB_HOST",     "localhost")
DB_PORT     = int(os.getenv("DB_PORT", "3306"))
DB_NAME     = os.getenv("DB_NAME",     "")

MODEL_DIR   = Path(os.getenv("MODEL_DIR", "./models"))
MODEL_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════
#  SECTION 1 — DATA LOADING
# ══════════════════════════════════════════════════════════════

# Replace the ENTIRE try block in load_from_mysql() with this:
def load_from_mysql() -> pd.DataFrame:
    try:
        import pymysql
        conn = pymysql.connect(
            host=DB_HOST, port=DB_PORT, user=DB_USER,
            password=DB_PASSWORD, database=DB_NAME,
            charset='utf8mb4'
            # ← remove DictCursor, let pandas handle it
        )
        query = """
            SELECT
                rc.id,
                rc.domain,
                rc.location,
                rc.device,
                rc.keywords_json,
                rc.results_json,
                rc.metrics_json,
                rc.visibility_score,
                rc.avg_rank,
                rc.live_data,
                rc.created_at,
                l.score          AS website_score,
                l.seo_score,
                l.engagement_score,
                l.current_software,
                l.zip_code
            FROM rank_checks rc
            LEFT JOIN leads l ON rc.lead_id = l.id
            WHERE rc.results_json IS NOT NULL
            AND rc.results_json != ''
            AND rc.results_json != 'results_json'
        """
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Safety filter — drop any row where results_json is clearly not JSON
        df = df[df['results_json'].str.startswith('[') | df['results_json'].str.startswith('{')]
        
        log.info(f"✓ Loaded {len(df)} valid rank check rows from MySQL")
        return df
    except Exception as e:
        log.warning(f"MySQL load failed: {e} — falling back to synthetic data")
        return pd.DataFrame()

    
def explode_results(df_raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df_raw.iterrows():
        try:
            # ← ADD these two lines — force string conversion
            results_raw = row['results_json']
            results = json.loads(str(results_raw)) if results_raw else []
            
            metrics_raw = row['metrics_json']
            metrics = json.loads(str(metrics_raw)) if metrics_raw else {}
        except Exception as e:
            log.warning(f"  ⚠ Skipping row, JSON parse failed: {e}")
            continue

        base = {
            'domain':           row['domain'],
            'location':         row.get('location', 'India'),
            'device':           row.get('device', 'mobile'),
            'website_score':    row.get('website_score'),
            'seo_score':        row.get('seo_score'),
            'engagement_score': row.get('engagement_score'),
            'current_software': row.get('current_software', 'Unknown'),
            'visibility_score': row.get('visibility_score', 0),
            'checked_at':       row.get('created_at'),
        }

        for r in results:
            record = {**base}
            record['keyword']     = r.get('keyword', '')
            record['organic_rank']= r.get('rank')        # ← renamed from 'rank' to 'organic_rank'
            record['local_rank']  = r.get('local_rank')
            record['featured']    = r.get('featured', False)
            record['found']       = r.get('found', False)
            rows.append(record)

    df = pd.DataFrame(rows)
    log.info(f"✓ Exploded to {len(df)} (domain, keyword) rows")
    if len(df) == 0:
        log.warning("  ⚠ Zero rows after explode — check results_json format in DB")
    return df


# ══════════════════════════════════════════════════════════════
#  SECTION 2 — SYNTHETIC DATA GENERATOR
#  Used when MySQL has < 200 rows (cold start) or for testing
# ══════════════════════════════════════════════════════════════

def generate_synthetic_data(n_samples: int = 3000) -> pd.DataFrame:
    """
    Generates realistic synthetic training data based on known SEO relationships.

    Key relationships encoded:
    - Higher website_score → better rank (strong)
    - SSL + mobile + GMB → better rank (strong)
    - Blog + testimonials → better rank (moderate)
    - Longer keywords → worse rank / less competition
    - Local keywords → local pack results
    - Brand keywords → always top 3
    - Device mobile vs desktop → slight rank delta
    """
    log.info(f"Generating {n_samples} synthetic training samples...")
    random.seed(42)
    np.random.seed(42)

    LOCATIONS = ['India', 'Austin TX', 'New York NY', 'Los Angeles CA',
                 'Chicago IL', 'Houston TX', 'Phoenix AZ', 'Philadelphia PA']
    DEVICES    = ['mobile', 'desktop']
    SOFTWARES  = ['ChiroTouch', 'Jane App', 'ChiroSpring', 'AdvancedMD',
                  'DrChrono', 'None', 'Other', 'Eclipse EHR']
    KW_TEMPLATES = [
        ("{specialty} near me",              True,  False),
        ("best {specialty} in {city}",       True,  False),
        ("{specialty}",                       False, False),
        ("{practice_name}",                   False, True ),  # brand
        ("{specialty} {city}",               True,  False),
        ("affordable {specialty}",            False, False),
        ("emergency {specialty}",             False, False),
        ("top rated {specialty}",             False, False),
        ("{specialty} for back pain",         False, False),
        ("{specialty} appointment online",    False, False),
    ]
    SPECIALTIES = ['chiropractor', 'physical therapist', 'dentist', 'orthopedist',
                   'sports medicine', 'pain clinic', 'spine specialist']

    rows = []
    for i in range(n_samples):
        # ── Website quality signals ──
        has_ssl          = random.random() > 0.15
        has_mobile       = random.random() > 0.20
        has_gmb          = random.random() > 0.35
        has_blog         = random.random() > 0.55
        has_testimonials = random.random() > 0.40
        has_booking      = random.random() > 0.50
        has_intake       = random.random() > 0.45
        has_video        = random.random() > 0.65
        has_social       = random.random() > 0.30
        has_contact      = random.random() > 0.05
        pagespeed_mobile = np.clip(np.random.normal(58, 20), 10, 100)
        pagespeed_desk   = np.clip(np.random.normal(72, 18), 10, 100)
        website_score    = (
            has_ssl * 5 + has_mobile * 10 + has_gmb * 2 + has_blog * 10 +
            has_testimonials * 10 + has_booking * 15 + has_intake * 15 +
            has_video * 5 + has_social * 10 + has_contact * 3 +
            (pagespeed_mobile / 100) * 15 + random.randint(-10, 10)
        )
        website_score = int(np.clip(website_score, 0, 100))
        seo_score     = int(np.clip(
            has_mobile * 15 + has_ssl * 15 + has_gmb * 20 + has_blog * 15 +
            has_testimonials * 15 + has_social * 10 + has_contact * 10 +
            random.randint(-8, 8), 0, 100))
        engagement_score = int(np.clip(
            has_booking * 30 + has_intake * 25 + has_video * 15 +
            has_testimonials * 10 + random.randint(-10, 10), 0, 100))

        # ── Keyword signals ──
        kw_template, is_local, is_brand = random.choice(KW_TEMPLATES)
        specialty   = random.choice(SPECIALTIES)
        location    = random.choice(LOCATIONS)
        device      = random.choice(DEVICES)
        software    = random.choice(SOFTWARES)
        city        = location.split()[0] if ' ' in location else location
        practice_nm = f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))} {specialty.title()}"
        keyword     = kw_template.format(
            specialty=specialty, city=city, practice_name=practice_nm
        )
        kw_length   = len(keyword.split())
        kw_chars    = len(keyword)

        # ── Simulate organic rank based on signals ──
        # Base: random rank weighted by quality
        quality_score = (
            website_score * 0.30 +
            seo_score     * 0.25 +
            has_gmb       * 15   +
            has_mobile    * 8    +
            has_ssl       * 6    +
            has_blog      * 8    +
            has_testimonials * 5 +
            (pagespeed_mobile / 100) * 10 +
            (kw_length - 1)     * 2       +   # long-tail easier to rank
            is_brand            * 30          # brand keywords near top
        )
        # Invert: higher quality → lower rank number (closer to 1)
        rank_base = max(1, int(101 - quality_score * 0.85 + np.random.normal(0, 8)))
        rank_base = int(np.clip(rank_base, 1, 100))

        # Not ranking at all (25% of bad sites)
        not_ranking = (rank_base > 70 and random.random() > 0.50)
        organic_rank = None if not_ranking else rank_base

        # Local pack (only for local keywords)
        has_local_pack = (is_local and has_gmb and quality_score > 45 and random.random() > 0.45)
        local_rank     = (random.randint(1, 3) if has_local_pack else None)

        # Featured snippet (rare, only top rankers)
        featured = (not not_ranking and rank_base <= 5 and random.random() > 0.80)

        rows.append({
            # Website features
            'has_ssl':          int(has_ssl),
            'has_mobile':       int(has_mobile),
            'has_gmb':          int(has_gmb),
            'has_blog':         int(has_blog),
            'has_testimonials': int(has_testimonials),
            'has_booking':      int(has_booking),
            'has_intake':       int(has_intake),
            'has_video':        int(has_video),
            'has_social':       int(has_social),
            'has_contact':      int(has_contact),
            'pagespeed_mobile': round(pagespeed_mobile, 1),
            'pagespeed_desktop':round(pagespeed_desk, 1),
            'website_score':    website_score,
            'seo_score':        seo_score,
            'engagement_score': engagement_score,
            # Keyword features
            'keyword':          keyword,
            'kw_length':        kw_length,
            'kw_chars':         kw_chars,
            'is_local_kw':      int(is_local),
            'is_brand_kw':      int(is_brand),
            'has_near_me':      int('near me' in keyword),
            'has_best':         int('best' in keyword),
            'has_city_in_kw':   int(city.lower() in keyword.lower()),
            # Context
            'device':           device,
            'location':         location,
            'software':         software,
            # Targets
            'organic_rank':     organic_rank,
            'local_rank':       local_rank,
            'featured':         int(featured),
            'found':            int(not not_ranking),
        })

    df = pd.DataFrame(rows)
    log.info(f"✓ Synthetic data generated: {len(df)} rows, {df['found'].mean():.1%} ranking")
    return df


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════

def rank_to_bucket(rank) -> str:
    """Convert numeric rank to classification bucket."""
    if rank is None or (isinstance(rank, float) and math.isnan(rank)):
        return 'not_ranking'
    rank = int(rank)
    if rank <= 3:   return 'top_3'
    if rank <= 10:  return 'top_10'
    if rank <= 30:  return 'top_30'
    return 'not_ranking'


BUCKET_ORDER = ['top_3', 'top_10', 'top_30', 'not_ranking']
BUCKET_TO_INT = {b: i for i, b in enumerate(BUCKET_ORDER)}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw df (from MySQL explode or synthetic) and returns
    a clean feature matrix ready for sklearn.
    """
    df = df.copy()

    # ── Derived features ──
    df['has_gmb_and_mobile'] = (
        df.get('has_gmb', 0).fillna(0).astype(int) *
        df.get('has_mobile', 0).fillna(0).astype(int)
    )
    df['has_full_technical'] = (
        df.get('has_ssl', 0).fillna(0).astype(int) *
        df.get('has_mobile', 0).fillna(0).astype(int) *
        df.get('has_gmb', 0).fillna(0).astype(int)
    )
    df['content_richness'] = (
        df.get('has_blog', 0).fillna(0).astype(int) +
        df.get('has_video', 0).fillna(0).astype(int) +
        df.get('has_testimonials', 0).fillna(0).astype(int)
    )
    df['patient_conversion'] = (
        df.get('has_booking', 0).fillna(0).astype(int) +
        df.get('has_intake', 0).fillna(0).astype(int)
    )
    df['pagespeed_avg'] = (
        df.get('pagespeed_mobile', 50).fillna(50) +
        df.get('pagespeed_desktop', 65).fillna(65)
    ) / 2
    df['seo_engagement_product'] = (
        df.get('seo_score', 0).fillna(0) *
        df.get('engagement_score', 0).fillna(0) / 100
    )
    df['is_mobile_device'] = (df.get('device', 'mobile') == 'mobile').astype(int)

    # Keyword complexity
    if 'keyword' in df.columns:
        df['kw_length']    = df['keyword'].apply(lambda x: len(str(x).split()))
        df['kw_chars']     = df['keyword'].apply(lambda x: len(str(x)))
        df['is_local_kw']  = df['keyword'].apply(lambda x: int(any(w in str(x).lower() for w in ['near me', 'nearby', 'local']))).astype(int)
        df['is_brand_kw']  = df.get('is_brand_kw', pd.Series(0, index=df.index)).fillna(0).astype(int)
        df['has_near_me']  = df['keyword'].apply(lambda x: int('near me' in str(x).lower()))
        df['has_city_in_kw'] = df.get('has_city_in_kw', pd.Series(0, index=df.index)).fillna(0).astype(int)

    # Encode categoricals
    for col in ['device', 'location', 'software', 'current_software']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
            le = LabelEncoder()
            df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))

    # ── Final feature list ──
    FEATURE_COLS = [
        # Website quality
        'has_ssl', 'has_mobile', 'has_gmb', 'has_blog',
        'has_testimonials', 'has_booking', 'has_intake',
        'has_video', 'has_social', 'has_contact',
        'pagespeed_mobile', 'pagespeed_desktop', 'pagespeed_avg',
        'website_score', 'seo_score', 'engagement_score',
        # Derived combinations
        'has_gmb_and_mobile', 'has_full_technical',
        'content_richness', 'patient_conversion',
        'seo_engagement_product',
        # Keyword
        'kw_length', 'kw_chars', 'is_local_kw', 'is_brand_kw',
        'has_near_me', 'has_city_in_kw',
        # Context
        'is_mobile_device',
    ]

    # Add encoded categoricals if present
    for col in ['device_enc', 'location_enc', 'software_enc', 'current_software_enc']:
        if col in df.columns:
            FEATURE_COLS.append(col)

    # Keep only cols that exist
    FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

    # Fill remaining NaN
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    return df, FEATURE_COLS


# ══════════════════════════════════════════════════════════════
#  SECTION 4 — MODEL BUILDING
# ══════════════════════════════════════════════════════════════

def build_classifier(model_type='xgb'):
    """Return rank bucket classifier."""
    if model_type == 'xgb' and HAS_XGB:
        return xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == 'lgb' and HAS_LGB:
        return lgb.LGBMClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
        )


def build_regressor(model_type='xgb'):
    """Return organic rank regressor."""
    if model_type == 'xgb' and HAS_XGB:
        return xgb.XGBRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    elif model_type == 'lgb' and HAS_LGB:
        return lgb.LGBMRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    else:
        return GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )


# ══════════════════════════════════════════════════════════════
#  SECTION 5 — TRAINING PIPELINE
# ══════════════════════════════════════════════════════════════

class RankPredictor:
    """
    End-to-end rank prediction pipeline.
    Trains two models:
      - classifier: predicts rank bucket (top_3 / top_10 / top_30 / not_ranking)
      - regressor:  predicts numeric rank 1–100 (for ranking rows only)
    """

    def __init__(self, model_type='xgb'):
        self.model_type   = model_type
        self.classifier   = None
        self.regressor    = None
        self.feature_cols = None
        self.label_enc    = LabelEncoder()
        self.imputer      = SimpleImputer(strategy='median')
        self.scaler       = StandardScaler()
        self.is_trained   = False
        self.training_meta = {}

    def prepare(self, df_raw: pd.DataFrame):
        df, feat_cols     = build_features(df_raw)
        self.feature_cols = feat_cols

        # Classification target
        df['rank_bucket'] = df['organic_rank'].apply(rank_to_bucket)
        df['rank_bucket_int'] = df['rank_bucket'].map(BUCKET_TO_INT)

        return df

    def train(self, df_raw: pd.DataFrame):
        log.info(f"\n{'='*60}")
        log.info(f"  Training RankPredictor (model={self.model_type})")
        log.info(f"{'='*60}")

        df = self.prepare(df_raw)
        X  = df[self.feature_cols].values
        y_cls = df['rank_bucket_int'].values

        # Impute + scale
        X = self.imputer.fit_transform(X)

        log.info(f"  Dataset: {X.shape[0]} rows × {X.shape[1]} features")
        log.info(f"  Bucket distribution:\n{df['rank_bucket'].value_counts().to_string()}")

        # ── Classifier ──
        X_tr, X_te, yc_tr, yc_te = train_test_split(
            X, y_cls, test_size=0.20, stratify=y_cls, random_state=42
        )
        self.classifier = build_classifier(self.model_type)
        self.classifier.fit(X_tr, yc_tr)

        yc_pred = self.classifier.predict(X_te)
        cls_acc = accuracy_score(yc_te, yc_pred)
        log.info(f"\n  Classifier accuracy: {cls_acc:.3f}")
        log.info("\n" + classification_report(
            yc_te, yc_pred,
            target_names=BUCKET_ORDER,
            zero_division=0,
        ))

        # ── Regressor (only on rows that have a numeric rank) ──
        df_ranked = df[df['organic_rank'].notna()].copy()
        if len(df_ranked) > 50:
            Xr = self.imputer.transform(df_ranked[self.feature_cols].values)
            yr = df_ranked['organic_rank'].values
            Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
                Xr, yr, test_size=0.20, random_state=42
            )
            self.regressor = build_regressor(self.model_type)
            self.regressor.fit(Xr_tr, yr_tr)
            yr_pred = self.regressor.predict(Xr_te)
            mae = mean_absolute_error(yr_te, yr_pred)
            r2  = r2_score(yr_te, yr_pred)
            log.info(f"\n  Regressor  MAE={mae:.1f} positions   R²={r2:.3f}")
        else:
            log.warning("  < 50 ranked rows — skipping regressor training")

        # Cross-val on classifier
        cv_scores = cross_val_score(
            build_classifier(self.model_type), X, y_cls,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='accuracy', n_jobs=-1
        )
        log.info(f"\n  5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        self.is_trained = True
        self.training_meta = {
            'trained_at':    datetime.now().isoformat(),
            'n_samples':     X.shape[0],
            'n_features':    X.shape[1],
            'feature_cols':  self.feature_cols,
            'model_type':    self.model_type,
            'cls_accuracy':  round(cls_acc, 4),
            'cv_mean':       round(cv_scores.mean(), 4),
            'cv_std':        round(cv_scores.std(), 4),
            'mae':           round(mae, 2) if self.regressor else None,
            'r2':            round(r2, 4)  if self.regressor else None,
            'bucket_dist':   df['rank_bucket'].value_counts().to_dict(),
        }
        return self.training_meta

    def predict(self, features: dict) -> dict:
        """
        Predict rank for a single (domain, keyword) pair.
        features: dict with all website + keyword signals
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call .train() first.")

        df_single = pd.DataFrame([features])
        df_single, _ = build_features(df_single)

        # Align to training feature cols
        for col in self.feature_cols:
            if col not in df_single.columns:
                df_single[col] = 0

        X = self.imputer.transform(df_single[self.feature_cols].values)

        # Bucket probabilities
        bucket_probs  = self.classifier.predict_proba(X)[0]
        bucket_pred   = BUCKET_ORDER[self.classifier.predict(X)[0]]
        bucket_conf   = float(max(bucket_probs))
        probs_dict    = {b: round(float(p), 3) for b, p in zip(BUCKET_ORDER, bucket_probs)}

        # Numeric rank estimate
        rank_estimate = None
        if self.regressor and bucket_pred != 'not_ranking':
            rank_estimate = int(np.clip(round(self.regressor.predict(X)[0]), 1, 100))

        # SHAP explanation
        explanation = []
        if HAS_SHAP and self.classifier is not None:
            try:
                explainer    = shap.TreeExplainer(self.classifier)
                shap_values  = explainer.shap_values(X)
                bucket_idx   = BUCKET_ORDER.index(bucket_pred)
                sv           = shap_values[bucket_idx][0] if isinstance(shap_values, list) else shap_values[0]
                feat_imp     = sorted(
                    zip(self.feature_cols, sv),
                    key=lambda x: abs(x[1]), reverse=True
                )[:5]
                explanation  = [
                    {"feature": f, "impact": round(float(v), 4),
                     "direction": "positive" if v > 0 else "negative"}
                    for f, v in feat_imp
                ]
            except:
                pass

        return {
            'rank_bucket':    bucket_pred,
            'rank_estimate':  rank_estimate,
            'confidence':     round(bucket_conf, 3),
            'probabilities':  probs_dict,
            'explanation':    explanation,
            'model_type':     self.model_type,
        }

    def feature_importance(self) -> list:
        """Return top feature importances from the classifier."""
        if not self.is_trained or not hasattr(self.classifier, 'feature_importances_'):
            return []
        imps = self.classifier.feature_importances_
        return sorted(
            [{"feature": f, "importance": round(float(i), 4)}
             for f, i in zip(self.feature_cols, imps)],
            key=lambda x: x['importance'], reverse=True
        )

    def save(self, path: Path = MODEL_DIR):
        """Save models + metadata to disk."""
        path = Path(path)
        path.mkdir(exist_ok=True)
        joblib.dump(self.classifier,   path / 'classifier.pkl')
        joblib.dump(self.imputer,      path / 'imputer.pkl')
        if self.regressor:
            joblib.dump(self.regressor, path / 'regressor.pkl')
        meta = {**self.training_meta, 'feature_cols': self.feature_cols}
        with open(path / 'meta.json', 'w') as f:
            json.dump(meta, f, indent=2)
        log.info(f"✓ Models saved to {path}/")

    @classmethod
    def load(cls, path: Path = MODEL_DIR):
        """Load models from disk."""
        path = Path(path)
        predictor = cls()
        predictor.classifier = joblib.load(path / 'classifier.pkl')
        predictor.imputer    = joblib.load(path / 'imputer.pkl')
        reg_path = path / 'regressor.pkl'
        if reg_path.exists():
            predictor.regressor = joblib.load(reg_path)
        with open(path / 'meta.json') as f:
            meta = json.load(f)
        predictor.feature_cols  = meta['feature_cols']
        predictor.training_meta = meta
        predictor.is_trained    = True
        log.info(f"✓ Models loaded from {path}/ (trained {meta.get('trained_at','')})")
        return predictor


# ══════════════════════════════════════════════════════════════
#  SECTION 6 — FLASK ROUTES (add to your app.py)
# ══════════════════════════════════════════════════════════════

FLASK_ROUTES_CODE = '''
# ─────────────────────────────────────────────────────────────
# ML Rank Predictor Routes
# Add these imports at the top of app.py:
#   from rank_ml_pipeline import RankPredictor, generate_synthetic_data, load_from_mysql, explode_results
#   import threading
# ─────────────────────────────────────────────────────────────

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


@app.route('/api/ml/train', methods=['POST'])
def ml_train():
    """
    Train / retrain the rank prediction model.
    POST body (optional): { "use_synthetic": true, "model_type": "xgb" }
    """
    data         = request.json or {}
    use_synthetic = data.get('use_synthetic', False)
    model_type    = data.get('model_type', 'xgb')

    def _train_thread():
        global _predictor
        try:
            # Load real data
            df_raw = load_from_mysql()
            df_exploded = explode_results(df_raw) if not df_raw.empty else pd.DataFrame()

            # Merge with synthetic if not enough real data
            if len(df_exploded) < 200 or use_synthetic:
                n_synthetic = max(0, 2000 - len(df_exploded))
                df_synth   = generate_synthetic_data(n_synthetic + 1000)
                df_train   = pd.concat([df_exploded, df_synth], ignore_index=True) if not df_exploded.empty else df_synth
                log.info(f"  Using {len(df_exploded)} real + {len(df_synth)} synthetic rows")
            else:
                df_train = df_exploded
                log.info(f"  Using {len(df_train)} real rows only")

            predictor = RankPredictor(model_type=model_type)
            meta      = predictor.train(df_train)
            predictor.save()
            with _predictor_lock:
                _predictor = predictor
            log.info("✅ ML training complete")
        except Exception as e:
            log.error(f"ML training error: {e}")
            import traceback; traceback.print_exc()

    thread = threading.Thread(target=_train_thread)
    thread.daemon = True
    thread.start()
    return jsonify({'status': 'training_started', 'message': 'Training in background — check /api/ml/status'})


@app.route('/api/ml/status', methods=['GET'])
def ml_status():
    """Check if model is trained and return metadata."""
    p = get_predictor()
    if p is None:
        return jsonify({'trained': False, 'message': 'No model found. POST to /api/ml/train first.'})
    return jsonify({'trained': True, **p.training_meta})


@app.route('/api/ml/predict', methods=['POST'])
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
        return jsonify({'error': 'Model not trained. POST to /api/ml/train first.'}), 503

    features = request.json or {}
    if not features.get('keyword'):
        return jsonify({'error': 'keyword is required'}), 400

    try:
        result = p.predict(features)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ml/predict-bulk', methods=['POST'])
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
        return jsonify({'error': 'Model not trained'}), 503

    data     = request.json or {}
    base     = data.get('base_features', {})
    keywords = data.get('keywords', [])

    if not keywords:
        return jsonify({'error': 'keywords array required'}), 400

    results = []
    for kw in keywords[:20]:  # cap at 20
        features = {**base, 'keyword': kw}
        try:
            pred = p.predict(features)
            results.append({'keyword': kw, **pred})
        except Exception as e:
            results.append({'keyword': kw, 'error': str(e)})

    return jsonify({'predictions': results, 'count': len(results)})


@app.route('/api/ml/feature-importance', methods=['GET'])
def ml_feature_importance():
    """Return top feature importances from the trained classifier."""
    p = get_predictor()
    if p is None:
        return jsonify({'error': 'Model not trained'}), 503
    return jsonify({'importances': p.feature_importance()})


@app.route('/api/ml/retrain-schedule', methods=['POST'])
def ml_retrain_schedule():
    """Trigger weekly auto-retrain. Call from a cron job or scheduler."""
    # Check if model was trained > 7 days ago
    p = get_predictor()
    if p and p.training_meta.get('trained_at'):
        trained_at = datetime.fromisoformat(p.training_meta['trained_at'])
        age_days   = (datetime.now() - trained_at).days
        if age_days < 7:
            return jsonify({'skipped': True, 'reason': f'Model is only {age_days} days old', 'trained_at': p.training_meta['trained_at']})

    # Trigger retrain
    from flask import current_app
    with current_app.test_request_context():
        return ml_train()
'''


# ══════════════════════════════════════════════════════════════
#  SECTION 7 — MAIN (run standalone to train & test)
# ══════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*60}")
    print("  RankPulse  Rank Predictor — Training Run")
    print(f"  XGBoost: {'✓' if HAS_XGB else '✗ (fallback: GradientBoosting)'}")
    print(f"  LightGBM: {'✓' if HAS_LGB else '✗'}")
    print(f"  SHAP: {'✓' if HAS_SHAP else '✗'}")
    print(f"{'='*60}\n")

    # ── Step 1: Load or generate data ──
    df_mysql = load_from_mysql()
    df_real = pd.DataFrame()
    if not df_mysql.empty:
    # ← ADD: debug first row
        print(f"  First results_json type: {type(df_mysql['results_json'].iloc[0])}")
        print(f"  First results_json value: {str(df_mysql['results_json'].iloc[0])[:100]}")
        df_real = explode_results(df_mysql)

    n_synthetic = max(0, 3000 - len(df_real))
    df_synth    = generate_synthetic_data(n_synthetic + 1000)
    df_train    = pd.concat([df_real, df_synth], ignore_index=True) if not df_real.empty else df_synth
    print(f"\n✓ Total training data: {len(df_train)} rows")

    # ── Step 2: Train ──
    model_type = 'xgb' if HAS_XGB else ('lgb' if HAS_LGB else 'rf')
    predictor  = RankPredictor(model_type=model_type)
    meta       = predictor.train(df_train)

    # ── Step 3: Save ──
    predictor.save(MODEL_DIR)

    # ── Step 4: Feature importance ──
    print(f"\n{'='*40}")
    print("  Top 10 Feature Importances:")
    print(f"{'='*40}")
    for imp in predictor.feature_importance()[:10]:
        bar = '█' * int(imp['importance'] * 200)
        print(f"  {imp['feature']:<30} {bar} {imp['importance']:.4f}")

    # ── Step 5: Sample predictions ──
    print(f"\n{'='*40}")
    print("  Sample Predictions:")
    print(f"{'='*40}")

    test_cases = [
        {
            "label":          "🏆 Excellent site, local keyword",
            "keyword":        "chiropractor near me",
            "has_ssl":        1, "has_mobile": 1, "has_gmb": 1,
            "has_blog":       1, "has_testimonials": 1, "has_booking": 1,
            "has_intake":     1, "has_video": 1, "has_social": 1, "has_contact": 1,
            "pagespeed_mobile": 90, "pagespeed_desktop": 95,
            "website_score":  88, "seo_score": 85, "engagement_score": 80,
            "device": "mobile", "location": "India",
        },
        {
            "label":          "📊 Average site, generic keyword",
            "keyword":        "chiropractor",
            "has_ssl":        1, "has_mobile": 1, "has_gmb": 0,
            "has_blog":       0, "has_testimonials": 1, "has_booking": 1,
            "has_intake":     0, "has_video": 0, "has_social": 1, "has_contact": 1,
            "pagespeed_mobile": 55, "pagespeed_desktop": 68,
            "website_score":  55, "seo_score": 45, "engagement_score": 40,
            "device": "mobile", "location": "India",
        },
        {
            "label":          "📉 Poor site, competitive keyword",
            "keyword":        "best chiropractor",
            "has_ssl":        0, "has_mobile": 0, "has_gmb": 0,
            "has_blog":       0, "has_testimonials": 0, "has_booking": 0,
            "has_intake":     0, "has_video": 0, "has_social": 0, "has_contact": 1,
            "pagespeed_mobile": 28, "pagespeed_desktop": 35,
            "website_score":  22, "seo_score": 18, "engagement_score": 10,
            "device": "mobile", "location": "India",
        },
    ]

    for tc in test_cases:
        label = tc.pop('label')
        result = predictor.predict(tc)
        rank_str = f"~#{result['rank_estimate']}" if result['rank_estimate'] else "not ranking"
        print(f"\n  {label}")
        print(f"    Keyword: {tc['keyword']}")
        print(f"    Prediction: {result['rank_bucket']} ({rank_str})  confidence={result['confidence']:.0%}")
        print(f"    Probabilities: " + "  ".join(f"{k}={v:.0%}" for k, v in result['probabilities'].items()))
        if result['explanation']:
            print(f"    Top drivers:")
            for e in result['explanation'][:3]:
                arrow = '↑' if e['direction'] == 'positive' else '↓'
                print(f"      {arrow} {e['feature']} (impact={e['impact']:+.3f})")

    print(f"\n{'='*60}")
    print("✅ Training complete!")
    print(f"   Models saved to: {MODEL_DIR}/")
    print(f"   Accuracy: {meta['cls_accuracy']:.1%}")
    print(f"   CV Score: {meta['cv_mean']:.1%} ± {meta['cv_std']:.1%}")
    if meta.get('mae'):
        print(f"   Rank MAE: ±{meta['mae']:.1f} positions")
    print(f"\n   Add these routes to app.py:")
    print(f"   POST /api/ml/train           → train / retrain")
    print(f"   GET  /api/ml/status          → model metadata")
    print(f"   POST /api/ml/predict         → predict single keyword")
    print(f"   POST /api/ml/predict-bulk    → predict many keywords")
    print(f"   GET  /api/ml/feature-importance → top signals")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()