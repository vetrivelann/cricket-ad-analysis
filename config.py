import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
FRAMES_DIR = os.path.join(BASE_DIR, "frames")
CHUNKS_DIR = os.path.join(BASE_DIR, "chunks")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

for folder in [UPLOAD_DIR, FRAMES_DIR, CHUNKS_DIR, REPORTS_DIR, CHROMA_DIR]:
    os.makedirs(folder, exist_ok=True)

# database settings
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "cricket_ads")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# video processing
FRAME_RATE = float(os.getenv("FRAME_RATE", "1.0"))
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# yolo settings
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_CONFIDENCE = float(os.getenv("YOLO_CONFIDENCE", "0.35"))

# COCO class -> brand label mapping
# YOLO detects generic objects. We map relevant ones to ad placement zones
# but NOT directly to brands. Brands are identified by OCR text only.
# This map is used to know WHICH COCO detections are interesting for analysis.
BRAND_LABEL_MAP = {
    "sports ball": "MRF",
    "tv":          "Scoreboard_Zone",
    "laptop":      "Overlay_Zone",
    "cell phone":  "Jio",
    "bottle":      "Pepsi",
    "cup":         "Boost",
    "clock":       "Paytm",
    "book":        "Dream11",
    "umbrella":    "PhonePe",
    "handbag":     "Byju's",
    "backpack":    "Tata",
    "tie":         "MPL",
    "bowl":        "CEAT",
    "person":      "Player_Detected",
}

# Known brand names for OCR matching
KNOWN_BRANDS = [
    "Boost", "Pepsi", "MRF", "CEAT", "Dream11",
    "Jio", "Paytm", "PhonePe", "Byju's", "Tata",
    "MPL", "Swiggy", "Unacademy", "VIVO", "Star Sports",
    "Thums Up", "Sprite", "Bournvita", "Coca-Cola",
]

# OCR-based brand correction: if OCR reads actual text, override YOLO label
BRAND_TEXT_PATTERNS = {
    "BOOST":       "Boost",
    "COCA":        "Coca-Cola",
    "PEPSI":       "Pepsi",
    "THUMS UP":    "Thums Up",
    "THUMSUP":     "Thums Up",
    "SPRITE":      "Sprite",
    "JIO":         "Jio",
    "DREAM11":     "Dream11",
    "DREAM 11":    "Dream11",
    "PAYTM":       "Paytm",
    "PHONEPE":     "PhonePe",
    "PHONE PE":    "PhonePe",
    "BYJUS":       "Byju's",
    "BYJU":        "Byju's",
    "CEAT":        "CEAT",
    "MRF":         "MRF",
    "MPL":         "MPL",
    "SWIGGY":      "Swiggy",
    "UNACADEMY":   "Unacademy",
    "VIVO":        "VIVO",
    "STAR SPORTS": "Star Sports",
    "TATA":        "Tata",
    "BOURNVITA":   "Bournvita",
}

# placement classification using vertical position ratios
PLACEMENT_RULES = [
    (0.0, 0.12, "scoreboard"),
    (0.0, 0.20, "overlay"),
    (0.80, 1.0, "boundary"),
    (0.20, 0.80, "jersey"),
]

# keywords for cricket event detection via OCR
EVENT_KEYWORDS = {
    "SIX":      "six",
    "FOUR":     "four",
    "OUT":      "wicket",
    "WICKET":   "wicket",
    "WIDE":     "wide",
    "NO BALL":  "no_ball",
    "BOUNDARY": "four",
}

# Gemini LLM settings (free tier)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Legacy OpenAI settings (optional fallback)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")

# padding when cutting video chunks (seconds)
CHUNK_PADDING = 1.0
