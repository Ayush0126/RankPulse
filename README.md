# RankPulse

RankPulse analyzes a healthcare practice's website and online presence, then scores it across SEO, performance, and patient-engagement criteria. It also checks live Google rankings for target keywords and surfaces nearby competitors.

**Live app:** https://rankpulse-73be.onrender.com
**Admin dashboard:** https://rankpulse-73be.onrender.com/admin

> Hosted on Render's free tier — the app may take a few seconds to wake up after a period of inactivity.

## Features

- **Website analysis** — crawls a practice's site and scores it (0–100) across 12 weighted criteria: mobile responsiveness, SSL, online booking, digital intake, patient forms, check-in systems, social media, testimonials, blog content, video content, contact info, and Google Business Profile presence.
- **AI-assisted scoring** — uses [Groq](https://groq.com) (LLM inference) alongside rule-based scraping signals to improve confidence on ambiguous criteria, with a weighted confidence model combining scraper, software-capability, and AI signals.
- **PageSpeed integration** — pulls live desktop/mobile performance scores from the Google PageSpeed Insights API and factors them into the overall score.
- **Rank checking** — checks where a domain ranks on Google for chosen keywords via [SerpApi](https://serpapi.com), with a deterministic fallback estimator when no API key is configured.
- **Competitor discovery** — finds nearby practices via OpenStreetMap/Overpass and Nominatim, scoring them the same way for side-by-side comparison.
- **ML rank prediction** — an optional trainable model (`rank.py`) that predicts keyword rank from website signals, trained on real data (when available) blended with synthetic data.
- **Admin dashboard** — view leads, correct AI scoring decisions, and export corrections as a fine-tuning dataset.
- **In-app AI assistant** — a chat widget (powered by Groq) that explains scores and answers questions about the analysis.

## Tech stack

| Layer | Technology |
|---|---|
| Backend | Flask, Flask-SQLAlchemy, Flask-CORS |
| Database | MySQL (production) with automatic SQLite fallback if no DB credentials are set |
| Scraping | BeautifulSoup, `curl_cffi` (TLS-fingerprint impersonation to avoid bot blocks) |
| AI | Groq (LLM analysis + chat assistant) |
| Search/Rank data | SerpApi (live Google rankings), Google PageSpeed Insights API |
| Geo/competitor data | OpenStreetMap Nominatim + Overpass API |
| ML | scikit-learn / XGBoost-based rank predictor (`rank.py`) |
| Frontend | Static HTML/React (CDN-loaded), served directly by Flask |
| Hosting | Render (Web Service, Python 3.11 runtime) |

## Project structure

```
.
├── app.py                  # Flask app: routes, models, analysis pipeline
├── rank.py                 # ML rank predictor (training + inference)
├── requirements.txt
├── runtime.txt             # Pins Python version for Render
├── frontend/
│   ├── index.html          # Public-facing site (diagnostic form, results, rankings)
│   └── admin_dashboard.html
└── models/                 # Trained ML model artifacts (not persisted on Render free tier)
```

## Environment variables

| Variable | Required | Notes |
|---|---|---|
| `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME` | No | If unset, the app falls back to a local SQLite file (`fallback.db`) — fine for testing, but not persisted across restarts on Render's free tier |
| `GROQ_API_KEY` | Yes, for AI features | Powers AI-assisted scoring and the chat assistant |
| `GROQ_MODEL` | Yes, for AI features | e.g. `llama-3.3-70b-versatile` |
| `SERPAPI_KEY` | No | Without it, rank checks use a deterministic estimate instead of live Google data |
| `GOOGLE_API_KEY`, `GOOGLE_CX` | No | Reserved for Google Custom Search integration |
| `PAGESPEED_API_KEY` | No | PageSpeed Insights works without a key at lower rate limits |

Set these directly as environment variables in Render's dashboard (no quotes, no `=`, just the raw value in the Value field) or in a local `.env` file for development.

## Running locally

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt

# create a .env file with the variables listed above

python app.py
```

The app runs at `http://localhost:5000` (public form) and `http://localhost:5000/admin` (admin dashboard).

## Deployment (Render)

- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app`
- **Runtime:** pinned via `runtime.txt` to Python 3.11.9 (newer versions can fail to build `pydantic_core` and similar packages from source on Render's build sandbox)
- Environment variables are configured in the Render dashboard under the service's **Environment** tab.

## API overview

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/analyze-website` | POST | Starts a background website analysis, returns an `analysis_id` |
| `/api/analysis-status/<id>` | GET | Poll for analysis progress/results |
| `/api/rank-check` | POST | Checks Google rankings for a domain across keywords |
| `/api/competitors` | POST | Finds and scores nearby competitor practices |
| `/api/chat` | POST | AI assistant chat endpoint |
| `/api/leads` | GET | List analyzed leads (admin) |
| `/api/leads/<id>/corrections` | GET/POST | View/submit human corrections to AI scoring |
| `/api/export-dataset` | GET | Export corrections as a JSONL fine-tuning dataset |
| `/api/ml/train` | POST | Train/retrain the rank prediction model |
| `/api/ml/predict` | POST | Predict rank for a keyword given website signals |
| `/api/health` | GET | Service health check |

## Notes & known limitations

- On Render's free tier, the filesystem is ephemeral — any trained ML model files (`models/classifier.pkl`) or the SQLite fallback database will be wiped on redeploy/restart. For persistence, connect a cloud MySQL database via the `DB_*` environment variables.
- The free tier also spins down after inactivity, so the first request after idling will be slow while the service cold-starts.
