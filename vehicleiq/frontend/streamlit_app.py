"""
VehicleIQ — Streamlit Web Interface
=====================================
Multimodal Vehicle Retrieval Dashboard:
  - Tab 1: 🔍 Text Search  — Dual-Encoder Hybrid Search (Dense SigLIP2 + Sparse BM25 + RRF)
  - Tab 2: 📸 Image Search — Visual Re-Identification (SigLIP2 Vision Encoder + Cosine Similarity)
  - Tab 3: 🚗 Vehicle Detail — Full metadata, attributes, and cloud CDN evidence gallery
"""

import io
import os
import textwrap
import requests
import streamlit as st
from PIL import Image

# ── Page configuration ────────────────────────────────────────────

st.set_page_config(
    page_title="VehicleIQ — Multimodal Vehicle Retrieval",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL = os.environ.get("VEHICLEIQ_API_URL", "http://localhost:8000")

# ── Custom CSS ────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* ── Global dark theme ── */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* ── Header ── */
    .viq-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.2rem 0 1.5rem 0;
        border-bottom: 1px solid #21262d;
        margin-bottom: 1.5rem;
    }
    .viq-logo {
        font-size: 1.8rem;
        font-weight: 700;
        color: #58a6ff;
        letter-spacing: -0.5px;
    }
    .viq-tagline {
        color: #8b949e;
        font-size: 0.95rem;
    }

    /* ── Cards ── */
    .viq-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    /* ── Badges ── */
    .rank-badge {
        display: inline-block;
        background: #1f6feb;
        color: #ffffff;
        font-weight: 600;
        font-size: 0.75rem;
        padding: 2px 7px;
        border-radius: 12px;
        margin-bottom: 4px;
    }
    .attr-badge {
        display: inline-block;
        background: #21262d;
        color: #8b949e;
        font-size: 0.75rem;
        padding: 2px 8px;
        border-radius: 6px;
        margin-right: 4px;
        margin-bottom: 3px;
        border: 1px solid #30363d;
    }
    .cdn-badge {
        display: inline-block;
        background: #238636;
        color: #ffffff;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 6px;
        border-radius: 4px;
        margin-bottom: 4px;
    }

    /* ── Score bar ── */
    .score-bar-bg {
        background: #21262d;
        border-radius: 4px;
        height: 6px;
        width: 100%;
        margin: 4px 0 8px 0;
        overflow: hidden;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 4px;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #161b22;
        border-radius: 8px;
        padding: 4px;
        gap: 6px;
        border: 1px solid #21262d;
    }
    .stTabs [data-baseweb="tab"] {
        color: #8b949e;
        border-radius: 6px;
        padding: 8px 18px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d;
        color: #58a6ff;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="viq-header">
        <div>
            <div class="viq-logo">🚗 VehicleIQ</div>
            <div class="viq-tagline">Multimodal Vehicle Retrieval Engine — SigLIP2 + BM25 + Cloud CDN</div>
        </div>
        <div style="text-align:right;">
            <span class="cdn-badge">☁️ Supabase Cloud Storage Active</span><br>
            <span style="color:#8b949e;font-size:0.8rem;">2,320 Vehicles Indexed</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ──────────────────────────────────

for key, default in {
    "search_results": [],
    "image_search_results": [],
    "selected_vehicle_id": None,
    "search_query": "",
    "last_search_query": "",
    "last_image_filename": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── API helpers ───────────────────────────────────────────────────

def api_get(path: str, timeout: int = 10) -> dict | None:
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("❌ Cannot connect to VehicleIQ backend. Is the FastAPI server running on port 8000?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, json_data: dict | None = None, timeout: int = 120) -> dict | None:
    try:
        r = requests.post(f"{API_URL}{path}", json=json_data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("❌ Cannot connect to VehicleIQ backend. Is the FastAPI server running on port 8000?")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post_image(path: str, file_bytes: bytes, filename: str, top_k: int = 10, timeout: int = 120) -> dict | None:
    try:
        files = {"file": (filename, file_bytes, "image/jpeg")}
        data = {"top_k": str(top_k)}
        r = requests.post(f"{API_URL}{path}", files=files, data=data, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        st.error("❌ Cannot connect to VehicleIQ backend. Is the FastAPI server running on port 8000?")
        return None
    except Exception as e:
        st.error(f"Image search error: {e}")
        return None


def fetch_image(url: str) -> Image.Image | None:
    """Fetch an image from cloud CDN or local server."""
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        pass
    return None


def score_color(score: float) -> str:
    if score >= 0.80:
        return "#3fb950"
    if score >= 0.65:
        return "#58a6ff"
    if score >= 0.50:
        return "#d29922"
    return "#8b949e"


# ── Render Vehicle Card Helper ────────────────────────────────────

def render_vehicle_grid(results: list[dict], search_mode: str = "text"):
    n_cols = 3
    for row_start in range(0, len(results), n_cols):
        row = results[row_start : row_start + n_cols]
        cols = st.columns(n_cols)

        for col, r in zip(cols, row):
            with col:
                vid_id = r.get("vehicle_id", "")
                dense_score = r.get("dense_score", r.get("score", 0.0))
                sparse_score = r.get("sparse_score", 0.0)
                rrf_score = r.get("hybrid_score", 0.0)
                thumb_url = r.get("thumbnail_url")
                descriptions = r.get("descriptions") or []

                # Thumbnail image (streamed from Supabase CDN)
                img = fetch_image(thumb_url) if thumb_url else None
                if img:
                    col.image(img, use_container_width=True)
                else:
                    col.markdown(
                        "<div style='background:#1c2128;height:140px;"
                        "display:flex;align-items:center;justify-content:center;"
                        "border-radius:8px;border:1px solid #21262d;'>"
                        "<span style='color:#484f58;font-size:2rem;'>🚗</span></div>",
                        unsafe_allow_html=True,
                    )

                bar_pct = max(0, min(100, int((dense_score + 1) / 2 * 100)))
                s_color = score_color(dense_score)

                desc_html = ""
                if descriptions:
                    desc_html = (
                        f"<div style='color:#8b949e;font-size:0.75rem;font-style:italic;"
                        f"margin-top:4px;border-left:2px solid #388bfd;padding-left:6px;'>"
                        f"&ldquo;{descriptions[0]}&rdquo;</div>"
                    )

                if search_mode == "text":
                    header_line = (
                        f'<span class="rank-badge">#{r["rank"]} RRF: {rrf_score:.5f}</span>'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:2px;">'
                        f'  <span style="color:{s_color};font-weight:600;font-size:0.8rem;">SigLIP2: {dense_score:.3f}</span>'
                        f'  <span style="color:#d29922;font-size:0.75rem;font-weight:500;">BM25: {sparse_score:.2f}</span>'
                        f'</div>'
                    )
                else:
                    header_line = (
                        f'<span class="rank-badge">#{r["rank"]} Visual Match</span>'
                        f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:2px;">'
                        f'  <span style="color:{s_color};font-weight:600;font-size:0.8rem;">Cosine Sim: {dense_score:.3f}</span>'
                        f'</div>'
                    )

                card_html = textwrap.dedent(f"""
                    <div style="padding:0 2px;">
                        {header_line}
                        <div class="score-bar-bg">
                            <div class="score-bar-fill" style="width:{bar_pct}%;background:{s_color};"></div>
                        </div>
                        <div style="margin-top:6px;line-height:1.8;">
                            <span class="attr-badge">🚗 {r.get('vehicle_type','—')}</span>
                            <span class="attr-badge">🎨 {r.get('color','—')}</span>
                            <span class="attr-badge">🚘 {r.get('body_style','—')}</span>
                        </div>
                        {desc_html}
                        <div style="color:#8b949e;font-size:0.73rem;margin-top:4px;">
                            ID: <code>{vid_id[:16]}…</code>
                        </div>
                    </div>
                """).strip()
                col.markdown(card_html, unsafe_allow_html=True)

                if col.button("View Detail →", key=f"sel_{search_mode}_{vid_id}_{r['rank']}", use_container_width=True):
                    st.session_state.selected_vehicle_id = vid_id
                    st.rerun()


# ── Tabs ──────────────────────────────────────────────────────────

tab_text, tab_image, tab_detail = st.tabs(
    ["🔍 Search by Text (Hybrid Dual-Encoder)", "📸 Search by Image (Visual Re-ID)", "🚗 Vehicle Detail"]
)

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — Text Search
# ═══════════════════════════════════════════════════════════════════

with tab_text:
    col_input, col_k, col_alpha, col_btn = st.columns([3.5, 1, 1.4, 1])

    with col_input:
        query = st.text_input(
            "Natural language description",
            value=st.session_state.search_query,
            placeholder="e.g. red sedan turning left at intersection, white pickup truck",
            label_visibility="collapsed",
            key="text_query_input",
        )

    with col_k:
        top_k = st.selectbox(
            "Top K",
            options=[6, 9, 12, 18, 30],
            index=1,
            label_visibility="collapsed",
            key="text_top_k",
        )

    with col_alpha:
        alpha = st.slider(
            "Dense / Sparse Balance",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="0.0 = Pure BM25 keyword matching | 1.0 = Pure SigLIP2 visual semantics | 0.5 = Balanced RRF",
            key="text_alpha_slider",
        )

    with col_btn:
        search_clicked = st.button("🔍 Search", use_container_width=True, key="search_text_btn")

    # Example query pills
    st.markdown("<span style='color:#8b949e;font-size:0.8rem;'>Try examples: </span>", unsafe_allow_html=True)
    example_cols = st.columns(6)
    examples = ["red sedan", "white pickup truck", "blue SUV", "black sedan", "bus", "silver hatchback"]
    for i, ex in enumerate(examples):
        if example_cols[i].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state.search_query = ex
            query = ex
            search_clicked = True

    if search_clicked and query.strip():
        st.session_state.search_query = query
        with st.spinner("Searching with Dual-Encoder Hybrid Engine (SigLIP2 + BM25)…"):
            result = api_post("/search", json_data={"query": query, "top_k": top_k, "alpha": alpha})
        if result:
            st.session_state.search_results = result.get("results", [])
            st.session_state.last_search_query = query

    results = st.session_state.search_results
    if results:
        last_q = st.session_state.last_search_query
        st.markdown(
            f"<div style='color:#8b949e;font-size:0.85rem;margin:0.8rem 0 1.2rem;'>"
            f"Showing <strong style='color:#c9d1d9;'>{len(results)}</strong> hybrid results for "
            f"<em>&quot;{last_q}&quot;</em> (Gallery: <strong>2,320</strong> vehicles, images from Supabase CDN)"
            f"</div>",
            unsafe_allow_html=True,
        )
        render_vehicle_grid(results, search_mode="text")
    elif st.session_state.last_search_query:
        st.markdown("<div style='text-align:center;color:#8b949e;padding:3rem;'>No results found.</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='text-align:center;color:#484f58;padding:4rem;font-size:1.1rem;'>"
            "Enter a natural-language description above or pick an example to search."
            "</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — Image Search (Visual Re-ID)
# ═══════════════════════════════════════════════════════════════════

with tab_image:
    st.markdown("### 📸 Search by Image (Visual Re-Identification)")
    st.markdown(
        "<div style='color:#8b949e;font-size:0.9rem;margin-bottom:1rem;'>"
        "Upload a vehicle photo or surveillance crop to find visually similar vehicles across the citywide gallery using pure SigLIP2 visual embeddings."
        "</div>",
        unsafe_allow_html=True,
    )

    col_upload_box, col_upload_preview = st.columns([1.4, 1])

    with col_upload_box:
        uploaded_file = st.file_uploader(
            "Upload vehicle image",
            type=["jpg", "jpeg", "png", "webp"],
            key="img_search_file",
            label_visibility="collapsed",
        )
        col_img_k, col_img_btn = st.columns([1, 1.5])
        with col_img_k:
            img_top_k = st.selectbox("Top K", options=[6, 9, 12, 18], index=1, key="img_top_k_select")
        with col_img_btn:
            st.write("")
            img_search_clicked = st.button("🔍 Find Matching Vehicles", use_container_width=True, key="btn_run_img_search")

    with col_upload_preview:
        if uploaded_file:
            st.image(uploaded_file, caption=f"Query Image: {uploaded_file.name}", use_container_width=True)

    if img_search_clicked and uploaded_file is not None:
        with st.spinner("Extracting SigLIP2 visual features and querying 2,320-vehicle gallery…"):
            res = api_post_image(
                "/search/image",
                file_bytes=uploaded_file.getvalue(),
                filename=uploaded_file.name,
                top_k=img_top_k,
            )
        if res:
            st.session_state.image_search_results = res.get("results", [])
            st.session_state.last_image_filename = uploaded_file.name

    img_results = st.session_state.image_search_results
    if img_results:
        st.markdown(
            f"<div style='color:#8b949e;font-size:0.85rem;margin:0.8rem 0 1.2rem;'>"
            f"Showing <strong style='color:#c9d1d9;'>{len(img_results)}</strong> visually similar vehicles for "
            f"<em>&quot;{st.session_state.last_image_filename}&quot;</em>"
            f"</div>",
            unsafe_allow_html=True,
        )
        render_vehicle_grid(img_results, search_mode="image")
    elif st.session_state.last_image_filename:
        st.markdown("<div style='text-align:center;color:#8b949e;padding:3rem;'>No matching vehicles found.</div>", unsafe_allow_html=True)
    else:
        st.markdown(
            "<div style='text-align:center;color:#484f58;padding:4rem;font-size:1.1rem;'>"
            "Upload an image of a car to find visually matching vehicles in the database."
            "</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — Vehicle Detail
# ═══════════════════════════════════════════════════════════════════

with tab_detail:
    st.markdown("### 🚗 Vehicle Details")

    selected_id = st.text_input(
        "Vehicle ID",
        value=st.session_state.selected_vehicle_id or "",
        placeholder="Enter Vehicle ID (e.g. b06c903c-a25d-45fe-b0d5-294f72e34023)",
        key="detail_id_input",
    )

    if selected_id:
        st.session_state.selected_vehicle_id = selected_id.strip()

    vehicle_id = st.session_state.selected_vehicle_id

    if vehicle_id:
        with st.spinner("Loading vehicle data…"):
            vehicle = api_get(f"/vehicles/{vehicle_id}")

        if vehicle:
            col_img, col_info = st.columns([1, 1.4])

            with col_img:
                st.markdown(f"**Vehicle ID:** ` {vehicle_id} `")
                crop_urls = vehicle.get("crop_urls", [])
                
                # Check if cloud CDN thumbnail is available
                first_crop = crop_urls[0] if crop_urls else f"https://imoizjabszamxkavarlg.supabase.co/storage/v1/object/public/vehicleiq-crops/crops/{vehicle_id}/frame_000001.jpg"
                main_img = fetch_image(first_crop)
                if main_img:
                    st.image(main_img, caption="Representative Crop (Cloud CDN)", use_container_width=True)
                else:
                    st.markdown(
                        "<div style='background:#1c2128;height:200px;display:flex;align-items:center;justify-content:center;border-radius:8px;border:1px solid #21262d;'>"
                        "<span style='color:#484f58;font-size:3rem;'>🚗</span></div>",
                        unsafe_allow_html=True,
                    )

            with col_info:
                v_data = vehicle.get("vehicle", {})
                db_vals = vehicle.get("db", {})
                rec = vehicle.get("json_record") or {}

                st.markdown("#### Detected Attributes")
                attr_cols = st.columns(2)
                attr_cols[0].markdown(f"**Type:** {vehicle.get('vehicle_type', 'vehicle').title()}")
                attr_cols[1].markdown(f"**Color:** {vehicle.get('color', 'unknown').title()}")

                st.markdown("---")
                st.markdown("#### Ground-Truth Natural Language Descriptions")
                descriptions = rec.get("descriptions") or []
                if descriptions:
                    for d in descriptions:
                        st.markdown(f"- *&ldquo;{d}&rdquo;*")
                else:
                    st.caption("No manual descriptions available.")

                st.markdown("---")
                st.markdown("#### Cloud Storage Asset URL")
                st.code(first_crop, language="text")
        else:
            st.warning(f"Vehicle `{vehicle_id}` not found.")
    else:
        st.markdown(
            "<div style='text-align:center;color:#484f58;padding:4rem;font-size:1.1rem;'>"
            "Select a vehicle from the Search tab or enter a Vehicle ID above."
            "</div>",
            unsafe_allow_html=True,
        )

# ── Footer ────────────────────────────────────────────────────────
st.markdown(
    "<div style='text-align:center;color:#484f58;font-size:0.8rem;padding:2rem 0 1rem;'>"
    "VehicleIQ · Multimodal Vehicle Retrieval · SigLIP2 + BM25 + Cloudflare/Supabase S3"
    "</div>",
    unsafe_allow_html=True,
)
