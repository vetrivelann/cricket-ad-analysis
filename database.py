"""
Database layer using SQLAlchemy.
Connects to PostgreSQL if available, otherwise falls back to a local SQLite file.
"""
import os
import logging
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Text, DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from config import DATABASE_URL, BASE_DIR

log = logging.getLogger(__name__)
Base = declarative_base()


# ---------- ORM Models ----------

class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(100), unique=True, nullable=False)
    team_a = Column(String(100), default="Team A")
    team_b = Column(String(100), default="Team B")
    match_type = Column(String(50), default="T20")
    location = Column(String(200), default="Unknown")
    match_date = Column(DateTime, default=datetime.utcnow)
    video_path = Column(Text, nullable=True)
    video_duration = Column(Float, default=0.0)
    status = Column(String(50), default="uploaded")
    created_at = Column(DateTime, default=datetime.utcnow)

    detections = relationship("BrandDetection", back_populates="match",
                              cascade="all, delete-orphan")
    aggregates = relationship("BrandAggregate", back_populates="match",
                              cascade="all, delete-orphan")


class BrandDetection(Base):
    __tablename__ = "brand_detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(100), ForeignKey("matches.match_id"), nullable=False)
    brand_name = Column(String(200), nullable=False)
    confidence = Column(Float, default=0.0)
    bbox = Column(JSON, nullable=True)
    timestamp = Column(Float, default=0.0)
    frame_index = Column(Integer, default=0)
    placement = Column(String(50), default="unknown")
    event = Column(String(50), default="none")
    detection_source = Column(String(20), default="yolo")
    created_at = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match", back_populates="detections")


class BrandAggregate(Base):
    __tablename__ = "brand_aggregates"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(String(100), ForeignKey("matches.match_id"), nullable=False)
    brand_name = Column(String(200), nullable=False)
    total_duration = Column(Float, default=0.0)
    visibility_ratio = Column(Float, default=0.0)
    detection_count = Column(Integer, default=0)
    avg_confidence = Column(Float, default=0.0)
    placement_distribution = Column(JSON, nullable=True)
    event_distribution = Column(JSON, nullable=True)
    chunk_paths = Column(JSON, nullable=True)
    start_time = Column(Float, default=0.0)
    end_time = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    match = relationship("Match", back_populates="aggregates")


# ---------- Engine setup with fallback ----------

def _create_engine():
    try:
        eng = create_engine(DATABASE_URL, pool_pre_ping=True)
        with eng.connect() as conn:
            from sqlalchemy import text
            conn.execute(text("SELECT 1"))
        log.info("Connected to PostgreSQL.")
        return eng
    except Exception:
        db_path = os.path.join(BASE_DIR, "cricket_ads.db")
        log.warning(f"PostgreSQL not available, using SQLite at {db_path}")
        return create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False}
        )


engine = _create_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db():
    Base.metadata.create_all(bind=engine)
    log.info("Database tables ready.")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------- CRUD helpers ----------

def create_match(db, match_id, video_path="", **kwargs):
    existing = db.query(Match).filter(Match.match_id == match_id).first()
    if existing:
        existing.video_path = video_path or existing.video_path
        existing.status = kwargs.get("status", existing.status)
        if "video_duration" in kwargs:
            existing.video_duration = kwargs["video_duration"]
        db.commit()
        db.refresh(existing)
        return existing

    record = Match(match_id=match_id, video_path=video_path, **kwargs)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def insert_detections(db, detections_list):
    objects = [BrandDetection(**d) for d in detections_list]
    db.bulk_save_objects(objects)
    db.commit()


def insert_aggregates(db, aggregates_list):
    objects = [BrandAggregate(**a) for a in aggregates_list]
    db.bulk_save_objects(objects)
    db.commit()


def get_match(db, match_id):
    return db.query(Match).filter(Match.match_id == match_id).first()


def get_detections(db, match_id):
    return db.query(BrandDetection).filter(BrandDetection.match_id == match_id).all()


def get_aggregates(db, match_id):
    return db.query(BrandAggregate).filter(BrandAggregate.match_id == match_id).all()


def get_all_matches(db):
    return db.query(Match).order_by(Match.created_at.desc()).all()


# run table creation when this module is imported
init_db()
