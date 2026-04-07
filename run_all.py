"""
CLI runner to launch different parts of the system.

Usage:
    python run_all.py                       # streamlit dashboard
    python run_all.py --api                 # fastapi server
    python run_all.py --both                # both at once
    python run_all.py --process video.mp4   # process a file directly
"""
import os
import sys
import argparse
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

HERE = os.path.dirname(os.path.abspath(__file__))


def start_streamlit():
    log.info("Starting Streamlit on http://localhost:8501")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        os.path.join(HERE, "app.py"),
        "--server.port=8501",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ])


def start_api():
    log.info("Starting FastAPI on http://localhost:8000")
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "api:app", "--host=0.0.0.0", "--port=8000", "--reload",
    ], cwd=HERE)


def start_both():
    import threading
    t = threading.Thread(target=start_api, daemon=True)
    t.start()
    log.info("FastAPI started in background.")
    start_streamlit()


def process_cli(video_path, fps):
    from utils import generate_match_id
    from processing import process_video
    from rag import store_detections_in_vectordb
    from report_generator import generate_html_report, generate_csv_report

    if not os.path.exists(video_path):
        log.error(f"File not found: {video_path}")
        sys.exit(1)

    mid = generate_match_id()
    log.info(f"Processing {video_path} as {mid}")

    result = process_video(video_path, mid, fps=fps, extract_video_chunks=True)
    log.info(f"Found {result['total_detections']} detections, {result['brands_found']} brands")

    if result["detections"]:
        try:
            store_detections_in_vectordb(result["detections"], mid)
            log.info("Search index updated.")
        except Exception as e:
            log.warning(f"Indexing failed: {e}")

    try:
        html = generate_html_report(mid)
        csv_f = generate_csv_report(mid)
        log.info(f"Reports: {html}, {csv_f}")
    except Exception as e:
        log.warning(f"Report generation failed: {e}")

    log.info("All done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cricket Ad Detection Runner")
    parser.add_argument("--api", action="store_true", help="Start FastAPI only")
    parser.add_argument("--both", action="store_true", help="Start Streamlit + FastAPI")
    parser.add_argument("--process", type=str, metavar="VIDEO", help="Process a video file")
    parser.add_argument("--fps", type=float, default=1.0, help="Sampling FPS")
    args = parser.parse_args()

    from database import init_db
    init_db()

    if args.process:
        process_cli(args.process, args.fps)
    elif args.api:
        start_api()
    elif args.both:
        start_both()
    else:
        start_streamlit()
