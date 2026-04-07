"""
Streamlit dashboard for the Cricket Ad Detection system.
Pages: Upload & Process, Analytics, AI Chatbot, Match History
"""
import os
import sys
import logging
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(
    page_title="Cricket Ad Analytics - Jio Hotstar",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import UPLOAD_DIR, CHUNKS_DIR
from utils import generate_match_id, ensure_dir, seconds_to_timestamp, format_duration

log = logging.getLogger(__name__)


# --------------- custom styling ---------------

def load_styles():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown li {
        color: #e0e0ff !important;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px; padding: 24px;
        color: white; text-align: center;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-4px); }
    .metric-card .val { font-size: 2.5rem; font-weight: 800; margin: 8px 0; }
    .metric-card .lbl {
        font-size: 0.9rem; opacity: 0.85;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .metric-card.green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 8px 32px rgba(17, 153, 142, 0.3);
    }
    .metric-card.orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        box-shadow: 0 8px 32px rgba(245, 87, 108, 0.3);
    }
    .metric-card.blue {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        box-shadow: 0 8px 32px rgba(79, 172, 254, 0.3);
    }

    .hero {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border-radius: 20px; padding: 40px 32px; margin-bottom: 24px;
        color: white; text-align: center;
        box-shadow: 0 12px 48px rgba(0,0,0,0.3);
    }
    .hero h1 {
        font-size: 2.2rem; font-weight: 800; margin-bottom: 8px;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero p { font-size: 1rem; opacity: 0.8; }

    .chat-user {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; padding: 12px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0; max-width: 80%; margin-left: auto;
    }
    .chat-bot {
        background: #f0f2f6; color: #1a1a2e;
        padding: 12px 18px; border-radius: 18px 18px 18px 4px;
        margin: 8px 0; max-width: 80%; border: 1px solid #e0e0e0;
    }

    .sec-head {
        font-size: 1.4rem; font-weight: 700; color: #1a1a2e;
        margin: 24px 0 12px 0; padding-bottom: 8px;
        border-bottom: 3px solid #667eea; display: inline-block;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)


def metric_card(label, value, variant=""):
    st.markdown(f"""
    <div class="metric-card {variant}">
        <div class="lbl">{label}</div>
        <div class="val">{value}</div>
    </div>
    """, unsafe_allow_html=True)


# --------------- session state ---------------

def init_session():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processing_result" not in st.session_state:
        st.session_state.processing_result = None
    if "current_match" not in st.session_state:
        st.session_state.current_match = None


# --------------- sidebar ---------------

def sidebar():
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Go to",
            ["Upload & Process", "Analytics Dashboard", "AI Chatbot", "Match History"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("### Settings")
        fps = st.slider("Frame rate (FPS)", 0.5, 5.0, 1.0, 0.5)
        do_chunks = st.checkbox("Extract video chunks", value=True)
        st.markdown("---")
        st.markdown(
            "<div style='text-align:center;opacity:0.5;font-size:0.8rem;'>"
            "Cricket Ad Analytics v1.0<br/>Jio Hotstar</div>",
            unsafe_allow_html=True,
        )
        return page, fps, do_chunks


# --------------- Page: Upload & Process ---------------

def page_upload(fps, do_chunks):
    st.markdown('<div class="sec-head">Upload & Process Video</div>', unsafe_allow_html=True)

    left, right = st.columns([2, 1])
    with left:
        uploaded = st.file_uploader(
            "Choose a cricket match video",
            type=["mp4", "avi", "mov", "mkv"],
        )
    with right:
        st.markdown("#### Match Info")
        team_a = st.text_input("Team A", "India")
        team_b = st.text_input("Team B", "Australia")
        match_type = st.selectbox("Type", ["T20", "ODI", "Test", "IPL"])
        venue = st.text_input("Venue", "Mumbai")

    if uploaded is not None:
        st.video(uploaded)

        if st.button("Start Processing", type="primary", use_container_width=True):
            mid = generate_match_id()
            st.session_state.current_match = mid

            # save the file
            save_dir = ensure_dir(os.path.join(UPLOAD_DIR, mid))
            vpath = os.path.join(save_dir, uploaded.name)
            with open(vpath, "wb") as f:
                f.write(uploaded.getbuffer())

            # create record
            from database import SessionLocal, create_match
            db = SessionLocal()
            try:
                create_match(db, mid, video_path=vpath,
                             team_a=team_a, team_b=team_b,
                             match_type=match_type, location=venue)
            finally:
                db.close()

            # run pipeline
            with st.spinner("Processing video, please wait..."):
                bar = st.progress(0, text="Starting up...")
                try:
                    bar.progress(10, text="Extracting frames and detecting brands...")
                    from processing import process_video
                    result = process_video(vpath, mid, fps=fps,
                                           extract_video_chunks=do_chunks)

                    bar.progress(80, text="Building search index...")
                    try:
                        from rag import store_detections_in_vectordb
                        store_detections_in_vectordb(result["detections"], mid)
                    except Exception as e:
                        log.warning(f"RAG indexing issue: {e}")

                    bar.progress(100, text="Done!")
                    st.session_state.processing_result = result

                    st.success(
                        f"Done! Found {result['total_detections']} detections "
                        f"across {result['brands_found']} brands."
                    )
                except Exception as e:
                    bar.progress(100, text="Failed")
                    st.error(f"Processing failed: {e}")
                    log.error(f"Pipeline error: {e}", exc_info=True)


# --------------- Page: Analytics ---------------

def page_analytics():
    st.markdown('<div class="sec-head">Analytics Dashboard</div>', unsafe_allow_html=True)

    from database import SessionLocal, get_all_matches, get_detections, get_aggregates

    db = SessionLocal()
    try:
        matches = get_all_matches(db)
    finally:
        db.close()

    if not matches:
        st.info("No matches processed yet. Upload a video first.")
        return

    options = {
        f"{m.match_id} ({m.team_a} vs {m.team_b})": m.match_id
        for m in matches
    }
    pick = st.selectbox("Select Match", list(options.keys()))
    chosen_id = options[pick]

    db = SessionLocal()
    try:
        match = next((m for m in matches if m.match_id == chosen_id), None)
        dets = get_detections(db, chosen_id)
        aggs = get_aggregates(db, chosen_id)
    finally:
        db.close()

    if not dets:
        st.warning("No detections found for this match.")
        return

    # top-level metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Detections", len(dets))
    with c2:
        metric_card("Brands Found", len(aggs), "green")
    with c3:
        dur = match.video_duration if match else 0
        metric_card("Video Duration", format_duration(dur), "orange")
    with c4:
        avg_c = sum(d.confidence for d in dets) / len(dets) if dets else 0
        metric_card("Avg Confidence", f"{avg_c:.0%}", "blue")

    st.markdown("<br/>", unsafe_allow_html=True)

    # brand visibility chart + detection share pie
    if aggs:
        left, right = st.columns(2)
        with left:
            st.markdown("#### Brand Visibility Duration")
            bar_data = [{"Brand": a.brand_name, "Duration (s)": a.total_duration,
                         "Visibility %": a.visibility_ratio} for a in aggs]
            fig = px.bar(bar_data, x="Brand", y="Duration (s)",
                         color="Visibility %", color_continuous_scale="Viridis",
                         text="Duration (s)")
            fig.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)",
                              font=dict(family="Inter"), height=400)
            fig.update_traces(texttemplate='%{text:.1f}s', textposition='outside')
            st.plotly_chart(fig, use_container_width=True)

        with right:
            st.markdown("#### Detection Share")
            counts = {}
            for d in dets:
                counts[d.brand_name] = counts.get(d.brand_name, 0) + 1
            fig2 = px.pie(values=list(counts.values()), names=list(counts.keys()),
                          color_discrete_sequence=px.colors.qualitative.Set3, hole=0.4)
            fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                               paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="Inter"), height=400)
            st.plotly_chart(fig2, use_container_width=True)

    # placement + event distribution
    st.markdown("#### Placement & Event Distribution")
    p1, p2 = st.columns(2)

    with p1:
        pl_counts = {}
        for d in dets:
            pl_counts[d.placement] = pl_counts.get(d.placement, 0) + 1
        fig3 = px.bar(x=list(pl_counts.keys()), y=list(pl_counts.values()),
                      labels={"x": "Placement", "y": "Count"},
                      color=list(pl_counts.keys()),
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig3.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Inter"), height=350)
        st.plotly_chart(fig3, use_container_width=True)

    with p2:
        ev_counts = {}
        for d in dets:
            if d.event and d.event != "none":
                ev_counts[d.event] = ev_counts.get(d.event, 0) + 1
        if ev_counts:
            fig4 = px.pie(values=list(ev_counts.values()), names=list(ev_counts.keys()),
                          title="Events", color_discrete_sequence=px.colors.qualitative.Bold)
            fig4.update_layout(plot_bgcolor="rgba(0,0,0,0)",
                               paper_bgcolor="rgba(0,0,0,0)",
                               font=dict(family="Inter"), height=350)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No cricket events detected in this match.")

    # brand x placement heatmap
    st.markdown("#### Brand x Placement Heatmap")
    bp = {}
    for d in dets:
        bp[(d.brand_name, d.placement)] = bp.get((d.brand_name, d.placement), 0) + 1

    if bp:
        brands = sorted(set(k[0] for k in bp))
        places = sorted(set(k[1] for k in bp))
        z = [[bp.get((b, p), 0) for p in places] for b in brands]
        fig5 = go.Figure(data=go.Heatmap(
            z=z, x=places, y=brands,
            colorscale="YlOrRd", texttemplate="%{z}",
        ))
        fig5.update_layout(height=max(300, len(brands) * 40),
                           plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Inter"))
        st.plotly_chart(fig5, use_container_width=True)

    # detection timeline
    st.markdown("#### Detection Timeline")
    tl_data = [{
        "Brand": d.brand_name, "Time (s)": d.timestamp,
        "Confidence": d.confidence, "Placement": d.placement,
    } for d in dets]

    if tl_data:
        fig6 = px.scatter(tl_data, x="Time (s)", y="Brand",
                          color="Placement", size="Confidence",
                          hover_data=["Confidence", "Placement"],
                          color_discrete_sequence=px.colors.qualitative.Vivid)
        unique_brands = set(d.brand_name for d in dets)
        fig6.update_layout(height=max(300, len(unique_brands) * 40),
                           plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(family="Inter"))
        st.plotly_chart(fig6, use_container_width=True)

    # raw data table
    st.markdown("#### Detection Records")
    df = pd.DataFrame([{
        "Brand": d.brand_name,
        "Confidence": f"{d.confidence:.2%}",
        "Timestamp": seconds_to_timestamp(d.timestamp),
        "Placement": d.placement,
        "Event": d.event,
        "Source": d.detection_source,
    } for d in dets])
    st.dataframe(df, use_container_width=True, height=400)

    # video chunks
    st.markdown("#### Video Chunks")
    found_any = False
    for a in aggs:
        if a.chunk_paths:
            paths = a.chunk_paths if isinstance(a.chunk_paths, list) else []
            for cp in paths:
                if os.path.exists(cp):
                    found_any = True
                    with st.expander(f"{a.brand_name} - {os.path.basename(cp)}"):
                        st.video(cp)
    if not found_any:
        st.info("No video chunks available.")


# --------------- Page: AI Chatbot ---------------

def page_chatbot():
    st.markdown('<div class="sec-head">AI Analytics Chatbot</div>', unsafe_allow_html=True)
    st.markdown("Ask questions about brand visibility, placements, or match events.")

    # quick query buttons
    st.markdown("**Quick queries:**")
    qcols = st.columns(3)
    quick = [
        "Which brand appeared most frequently?",
        "How many times did Pepsi appear during a six?",
        "Show placement distribution for all brands",
    ]
    for i, q in enumerate(quick):
        with qcols[i]:
            if st.button(q, key=f"q_{i}", use_container_width=True):
                st.session_state.chat_history.append({"role": "user", "content": q})
                try:
                    from rag import answer_query
                    ans = answer_query(q)
                except Exception as e:
                    ans = f"Error: {e}"
                st.session_state.chat_history.append({"role": "assistant", "content": ans})

    st.markdown("---")

    # display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bot">{msg["content"]}</div>',
                        unsafe_allow_html=True)

    # input box
    user_msg = st.chat_input("Ask about brand visibility, placements, events...")
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})
        with st.spinner("Thinking..."):
            try:
                from rag import answer_query
                ans = answer_query(user_msg)
            except Exception as e:
                ans = f"Error: {e}"
        st.session_state.chat_history.append({"role": "assistant", "content": ans})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# --------------- Page: Match History ---------------

def page_history():
    st.markdown('<div class="sec-head">Match History</div>', unsafe_allow_html=True)

    from database import SessionLocal, get_all_matches
    db = SessionLocal()
    try:
        matches = get_all_matches(db)
    finally:
        db.close()

    if not matches:
        st.info("No matches recorded yet.")
        return

    for m in matches:
        status_color = {"uploaded": "#fff3cd", "processing": "#cce5ff",
                        "completed": "#d4edda", "failed": "#f8d7da"}
        bg = status_color.get(m.status, "#f0f0f0")

        with st.expander(f"{m.team_a} vs {m.team_b} | {m.match_type} | {m.status.upper()}"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.write(f"**Match ID:** `{m.match_id}`")
                st.write(f"**Location:** {m.location}")
            with c2:
                st.write(f"**Duration:** {format_duration(m.video_duration)}")
                st.write(f"**Date:** {m.created_at}")
            with c3:
                st.write(f"**Status:** {m.status}")
                if m.video_path and os.path.exists(m.video_path):
                    st.write("Video file present")
                else:
                    st.write("Video file missing")


# --------------- Main ---------------

def main():
    load_styles()
    init_session()

    st.markdown("""
    <div class="hero">
        <h1>Cricket Ad Detection & Analytics</h1>
        <p>AI-Powered Brand Visibility Analysis for Jio Hotstar Cricket Broadcasts</p>
    </div>
    """, unsafe_allow_html=True)

    page, fps, do_chunks = sidebar()

    if page == "Upload & Process":
        page_upload(fps, do_chunks)
    elif page == "Analytics Dashboard":
        page_analytics()
    elif page == "AI Chatbot":
        page_chatbot()
    elif page == "Match History":
        page_history()


if __name__ == "__main__":
    main()
