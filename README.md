# 🏏 AI-Powered Cricket Advertisement Detection & Analytics System

> **End-to-end AI pipeline** for detecting brand advertisements in cricket match broadcasts, computing visibility analytics, and enabling stakeholder queries via RAG-powered conversational AI.

Built for **Jio Hotstar** cricket broadcast analytics.

---

## 📁 Project Structure

```
project_2/
│
├── config.py               # Central configuration (paths, models, DB, brands)
├── database.py             # SQLAlchemy ORM models + CRUD (PostgreSQL / SQLite)
├── detection.py            # YOLOv8 + OCR brand detection, placement & event classification
├── processing.py           # Video pipeline: frames → detection → aggregation → chunks
├── rag.py                  # ChromaDB vector store + RAG query engine
├── report_generator.py     # HTML & CSV report generation
├── api.py                  # FastAPI REST API (upload, process, query)
├── app.py                  # Streamlit dashboard (upload, charts, chatbot)
├── utils.py                # Shared utilities (timestamps, intervals, IDs)
├── run_all.py              # CLI runner for all services
│
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (edit for your setup)
├── .env.example            # Env template with documentation
├── .gitignore              # Git ignore rules
│
├── uploads/                # Uploaded video files
├── frames/                 # Extracted frames
├── chunks/                 # Brand-specific video clips
│   └── {brand}/{match_id}/
├── reports/                # Generated HTML/CSV reports
│   └── {match_id}/
└── chroma_db/              # ChromaDB vector store persistence
```

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **PostgreSQL** (optional — falls back to SQLite automatically)
- **Tesseract OCR** (optional — uses mock if not installed)
- **FFmpeg** (optional — for moviepy chunk extraction)

### 2. Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
# Copy the example env file
copy .env.example .env       # Windows
# cp .env.example .env       # Linux/Mac

# Edit .env with your PostgreSQL credentials (or leave defaults for SQLite)
```

### 4. Run the Application

#### Option A: Streamlit Dashboard (recommended)
```bash
streamlit run app.py
# or
python run_all.py
```
Opens at **http://localhost:8501**

#### Option B: FastAPI Server
```bash
python run_all.py --api
# or
uvicorn api:app --reload --port 8000
```
Opens at **http://localhost:8000/docs** (Swagger UI)

#### Option C: Both Services
```bash
python run_all.py --both
```
- Streamlit: http://localhost:8501
- FastAPI: http://localhost:8000

#### Option D: CLI Processing
```bash
python run_all.py --process path/to/video.mp4 --fps 1.0
```

---

## 🧩 Module Details

### 1. 🎥 Video Processing (`processing.py`)
- Accepts video Upload (MP4, AVI, MOV, MKV)
- Extracts frames using **OpenCV** at configurable FPS (default: 1 FPS)
- Resizes frames to 1280×720 for consistent detection

### 2. 🧠 Object Detection (`detection.py`)
- **YOLOv8** (ultralytics) for brand/logo detection
- Uses pretrained COCO model with brand label mapping
- Auto-downloads `yolov8n.pt` on first run
- Falls back to **mock detections** if YOLO is unavailable
- Returns: `brand_name`, `confidence`, `bounding_box`, `timestamp`

### 3. 🔤 OCR Support (`detection.py`)
- **pytesseract** for text-based brand detection
- Preprocesses frames (grayscale, blur, threshold)
- Matches against known brand list
- Falls back to mock OCR if Tesseract is not installed

### 4. 📍 Placement Classification (`detection.py`)
- Rule-based classification using bounding box Y-position:
  - **Scoreboard**: top 12% of frame
  - **Overlay**: top 20% of frame
  - **Boundary**: bottom 20% of frame
  - **Jersey**: middle 60% of frame

### 5. ⚡ Event Detection (`detection.py`)
- Detects cricket events: **SIX**, **FOUR**, **OUT/WICKET**, **WIDE**, **NO BALL**
- Uses OCR on scoreboard region (top 15% of frame)
- Falls back to mock random events

### 6. ⏱️ Timestamp Aggregation (`processing.py`)
- Merges continuous detections within 2-second gap
- Calculates: `start_time`, `end_time`, `total_duration`
- Computes **visibility ratio** (brand duration / match duration)

### 7. 🎬 Video Chunk Extraction (`processing.py`)
- Extracts brand-specific video clips using **moviepy** (or OpenCV fallback)
- Saves to: `chunks/{brand}/{match_id}/{start_end}.mp4`
- Adds 1-second padding to avoid clipping

### 8. 🗄️ Database (`database.py`)
- **PostgreSQL** via SQLAlchemy (auto-falls back to SQLite)
- Tables:
  - `matches` — match metadata (teams, type, location, video path)
  - `brand_detections` — per-frame detections with bbox, placement, event
  - `brand_aggregates` — aggregated metrics per brand per match

### 9. 📊 Aggregation (`processing.py`)
- Per-brand metrics:
  - Total display duration (seconds)
  - Visibility ratio (%)
  - Detection count
  - Average confidence
  - Placement distribution (jersey: N, boundary: M, ...)
  - Event distribution (six: N, four: M, ...)

### 10. 🤖 RAG System (`rag.py`)
- **ChromaDB** vector store for detection embeddings
- **sentence-transformers** (`all-MiniLM-L6-v2`) for embedding generation
- Supports **OpenAI GPT** for natural language answers
- Falls back to keyword-based retrieval if no API key
- Example queries:
  - *"How many times did Pepsi appear during sixes?"*
  - *"Which brand had the most boundary exposure?"*

### 11. 🌐 Streamlit Frontend (`app.py`)
- **Upload & Process**: Video upload with match metadata, progress bar
- **Analytics Dashboard**:
  - KPI cards (detections, brands, duration, confidence)
  - Brand visibility bar chart
  - Detection share pie chart
  - Placement & event distribution charts
  - Brand × Placement heatmap
  - Detection timeline scatter plot
  - Full detection data table
  - Video chunk viewer
- **AI Chatbot**: RAG-powered Q&A with quick query buttons
- **Match History**: Browse all processed matches

### 12. 🔌 FastAPI API (`api.py`)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/upload` | POST | Upload video + create match |
| `/process/{match_id}` | POST | Trigger async processing |
| `/matches` | GET | List all matches |
| `/results/{match_id}` | GET | Get detections + aggregates |
| `/query` | POST | RAG natural-language query |
| `/chunks/{brand}/{match_id}/{file}` | GET | Serve video chunk |

### 13. 📄 Report Generator (`report_generator.py`)
- **HTML reports**: Premium styled reports with brand summary tables
- **CSV exports**: Detection details and aggregate data
- Reports saved to `reports/{match_id}/`

---

## 🔧 Configuration

All settings are in `config.py` and loaded from `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | localhost | PostgreSQL host |
| `DB_PORT` | 5432 | PostgreSQL port |
| `DB_NAME` | cricket_ads | Database name |
| `DB_USER` | postgres | DB username |
| `DB_PASSWORD` | postgres | DB password |
| `FRAME_RATE` | 1.0 | Frames per second to extract |
| `YOLO_MODEL` | yolov8n.pt | YOLO model file |
| `YOLO_CONFIDENCE` | 0.35 | Min detection confidence |
| `OPENAI_API_KEY` | (empty) | OpenAI key for RAG |
| `LLM_MODEL` | gpt-3.5-turbo | LLM model name |

---

## 📦 Tech Stack

| Layer | Technology |
|-------|-----------|
| Detection | YOLOv8 (ultralytics), pytesseract |
| Video | OpenCV, moviepy, FFmpeg |
| Database | PostgreSQL / SQLite (SQLAlchemy) |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers |
| LLM | OpenAI GPT (optional) |
| Dashboard | Streamlit, Plotly |
| API | FastAPI, Uvicorn |
| Deployment | Docker-ready, CLI runner |

---

## 🧪 Testing Without GPU/Models

The system works **fully offline** with mock detections:
- If YOLO model fails to load → mock brand detections are generated
- If Tesseract is not installed → mock OCR results are generated
- If PostgreSQL is not running → SQLite is used automatically
- If no OpenAI key → keyword-based RAG answers are returned

This lets you **test the entire pipeline** without any external dependencies.

---

## 📝 License

Internal project for Jio Hotstar cricket broadcast analytics.
