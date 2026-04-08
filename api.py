"""
FastAPI endpoints for the Cricket Ad Detection system.
Enhanced with AI insights and Gemini-powered analytics.
"""
import os
import shutil
import logging

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from config import UPLOAD_DIR, CHUNKS_DIR
from utils import generate_match_id, ensure_dir

log = logging.getLogger(__name__)

app = FastAPI(
    title="Cricket Ad Detection API",
    description="AI-powered brand advertisement detection and analytics for cricket broadcasts",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    n_results: Optional[int] = 10


@app.get("/health")
def health():
    gemini_ok = False
    try:
        from gemini_client import is_available
        gemini_ok = is_available()
    except Exception:
        pass
    return {
        "status": "ok",
        "service": "cricket-ad-detection",
        "version": "2.0.0",
        "gemini_available": gemini_ok,
    }


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    team_a: str = Form("Team A"),
    team_b: str = Form("Team B"),
    match_type: str = Form("T20"),
    location: str = Form("Unknown"),
):
    from database import SessionLocal, create_match

    if not file.filename:
        raise HTTPException(400, "No file provided")

    mid = generate_match_id()
    save_dir = ensure_dir(os.path.join(UPLOAD_DIR, mid))
    video_path = os.path.join(save_dir, file.filename)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    db = SessionLocal()
    try:
        create_match(db, mid, video_path=video_path,
                     team_a=team_a, team_b=team_b,
                     match_type=match_type, location=location, status="uploaded")
    finally:
        db.close()

    return {"match_id": mid, "video_path": video_path, "status": "uploaded",
            "message": f"Uploaded. POST /process/{mid} to start analysis."}


@app.post("/process/{match_id}")
async def process_match(match_id: str, background_tasks: BackgroundTasks,
                        fps: float = 1.0, extract_chunks: bool = True):
    from database import SessionLocal, get_match

    db = SessionLocal()
    try:
        match = get_match(db, match_id)
        if not match:
            raise HTTPException(404, "Match not found")
        if not match.video_path or not os.path.exists(match.video_path):
            raise HTTPException(400, "Video file missing from disk")
        vpath = match.video_path
    finally:
        db.close()

    background_tasks.add_task(_process_bg, vpath, match_id, fps, extract_chunks)
    return {"match_id": match_id, "status": "processing",
            "message": "Processing started. Check /results/{match_id} when done."}


def _process_bg(video_path, match_id, fps, do_chunks):
    from processing import process_video
    from rag import store_detections_in_vectordb
    try:
        result = process_video(video_path, match_id, fps=fps,
                               extract_video_chunks=do_chunks)
        if result["detections"]:
            store_detections_in_vectordb(result["detections"], match_id)
    except Exception as err:
        log.error(f"Processing failed for {match_id}: {err}", exc_info=True)
        from database import SessionLocal, create_match
        db = SessionLocal()
        try:
            create_match(db, match_id, status="failed")
        finally:
            db.close()


@app.get("/matches")
def list_matches():
    from database import SessionLocal, get_all_matches
    db = SessionLocal()
    try:
        matches = get_all_matches(db)
        return [{
            "match_id": m.match_id, "team_a": m.team_a, "team_b": m.team_b,
            "match_type": m.match_type, "status": m.status,
            "video_duration": m.video_duration, "created_at": str(m.created_at),
        } for m in matches]
    finally:
        db.close()


@app.get("/results/{match_id}")
def get_results(match_id: str):
    from database import SessionLocal, get_match, get_detections, get_aggregates

    db = SessionLocal()
    try:
        match = get_match(db, match_id)
        if not match:
            raise HTTPException(404, "Match not found")

        dets = get_detections(db, match_id)
        aggs = get_aggregates(db, match_id)

        return {
            "match_id": match_id,
            "status": match.status,
            "video_duration": match.video_duration,
            "total_detections": len(dets),
            "brands_found": len(aggs),
            "detections": [{
                "brand_name": d.brand_name, "confidence": d.confidence,
                "timestamp": d.timestamp, "placement": d.placement,
                "event": d.event, "source": d.detection_source,
            } for d in dets],
            "aggregates": [{
                "brand_name": a.brand_name,
                "total_duration": a.total_duration,
                "visibility_ratio": a.visibility_ratio,
                "detection_count": a.detection_count,
                "avg_confidence": a.avg_confidence,
                "placement_distribution": a.placement_distribution,
                "event_distribution": a.event_distribution,
                "chunk_paths": a.chunk_paths,
            } for a in aggs],
        }
    finally:
        db.close()


@app.post("/query")
def rag_query(req: QueryRequest):
    from rag import answer_query
    if not req.question.strip():
        raise HTTPException(400, "Empty question")
    return {"question": req.question, "answer": answer_query(req.question)}


@app.get("/insights/{match_id}")
def get_insights(match_id: str):
    """Generate AI-powered insights for a processed match."""
    from database import SessionLocal, get_match, get_detections, get_aggregates
    from insights import generate_insights

    db = SessionLocal()
    try:
        match = get_match(db, match_id)
        if not match:
            raise HTTPException(404, "Match not found")

        dets = get_detections(db, match_id)
        aggs = get_aggregates(db, match_id)

        if not dets:
            raise HTTPException(400, "No detections found for this match")

        det_dicts = [{
            "brand_name": d.brand_name, "confidence": d.confidence,
            "timestamp": d.timestamp, "placement": d.placement,
            "event": d.event, "detection_source": d.detection_source,
        } for d in dets]

        agg_dicts = [{
            "brand_name": a.brand_name,
            "total_duration": a.total_duration,
            "visibility_ratio": a.visibility_ratio,
            "detection_count": a.detection_count,
            "avg_confidence": a.avg_confidence,
            "placement_distribution": a.placement_distribution,
            "event_distribution": a.event_distribution,
        } for a in aggs]

        match_info = {
            "team_a": match.team_a,
            "team_b": match.team_b,
            "match_type": match.match_type,
            "location": match.location,
        }

        result = generate_insights(det_dicts, agg_dicts, match_info)
        return {
            "match_id": match_id,
            "source": result.get("source", "unknown"),
            "insights": result.get("content", ""),
            "stats": result.get("stats", {}),
        }
    finally:
        db.close()


@app.get("/report/{match_id}")
def generate_report(match_id: str):
    """Generate and download an HTML report for a match."""
    from report_generator import generate_html_report
    try:
        path = generate_html_report(match_id)
        return FileResponse(path, media_type="text/html",
                            filename=os.path.basename(path))
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Report generation failed: {e}")


@app.get("/chunks/{brand}/{match_id}/{filename}")
def serve_chunk(brand: str, match_id: str, filename: str):
    path = os.path.join(CHUNKS_DIR, brand, match_id, filename)
    if not os.path.exists(path):
        raise HTTPException(404, "Chunk not found")
    return FileResponse(path, media_type="video/mp4")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
