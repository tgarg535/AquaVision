"""
Aqua Vision - Underwater Marine Species Detection Web Application
A Streamlit web application integrating CLAHE enhancement, YOLO11 inference, and SQLite logging.
"""

import base64
import os
import io
import json
from typing import Optional
from PIL import Image
import pandas as pd
import streamlit as st

# Import custom modular engines
from processing import apply_clahe
from detector import find_model_path, load_yolo_model, run_inference
from database import init_db, log_detection, get_detection_history, clear_history

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Aqua Vision: Underwater Marine Species Detector",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Background Wallpaper Styling (40-50% transparency)
# ---------------------------------------------------------
def get_wallpaper_path() -> Optional[str]:
    """Finds wallpaper image in the Pictures directory."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "Pictures", "wp2229213-sea-animal-wallpapers.jpg"),
        os.path.join(base_dir, "pictures", "wp2229213-sea-animal-wallpapers.jpg"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    # Fallback to any image file in Pictures directory
    for folder in ["Pictures", "pictures"]:
        p_dir = os.path.join(base_dir, folder)
        if os.path.isdir(p_dir):
            for fname in os.listdir(p_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    return os.path.join(p_dir, fname)
    return None


def get_base64_image(image_path: str) -> str:
    """Encodes a local image file to base64 string."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


def apply_custom_styles():
    """Applies ocean glassmorphism styling and transparent background."""
    wallpaper_path = get_wallpaper_path()
    bg_css = ""
    if wallpaper_path and os.path.isfile(wallpaper_path):
        b64_img = get_base64_image(wallpaper_path)
        bg_css = f"""
        .stApp {{
            background-color: #071322;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-image: url("data:image/jpeg;base64,{b64_img}");
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            opacity: 0.45; /* 45% transparency / opacity */
            z-index: 0;
            pointer-events: none;
        }}
        """
    else:
        bg_css = """
        .stApp {
            background: linear-gradient(135deg, #071322 0%, #0d2744 100%);
        }
        """

    st.markdown(
        f"""
        <style>
        {bg_css}

        /* Elevate all streamlit content above background */
        .stApp > header, .stApp > div, .main {{
            position: relative;
            z-index: 1;
        }}

        /* Sidebar Styling */
        [data-testid="stSidebar"] {{
            background: rgba(6, 18, 36, 0.88) !important;
            backdrop-filter: blur(14px);
            -webkit-backdrop-filter: blur(14px);
            border-right: 1px solid rgba(56, 189, 248, 0.2);
        }}

        /* Header / Banner styling */
        .aqua-title-container {{
            background: rgba(10, 29, 58, 0.75);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 16px;
            padding: 22px 28px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
        }}

        .aqua-title {{
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38bdf8, #00f2fe, #a7f3d0);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        }}

        .aqua-subtitle {{
            font-size: 1.05rem;
            color: #bae6fd;
            font-weight: 400;
        }}

        /* Cards & Metric containers */
        div[data-testid="stMetric"] {{
            background: rgba(10, 27, 53, 0.8) !important;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(56, 189, 248, 0.25) !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
        }}

        div[data-testid="stMetricLabel"] > div {{
            color: #7dd3fc !important;
            font-weight: 600 !important;
            font-size: 0.95rem !important;
        }}

        div[data-testid="stMetricValue"] > div {{
            color: #ffffff !important;
            font-size: 1.9rem !important;
            font-weight: 700 !important;
        }}

        /* Glass container helper */
        .glass-card {{
            background: rgba(10, 27, 53, 0.75);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(56, 189, 248, 0.25);
            border-radius: 14px;
            padding: 18px 22px;
            margin-bottom: 20px;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Initialize Database
# ---------------------------------------------------------
init_db()

# Apply CSS
apply_custom_styles()


# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Aqua Vision Settings")
    st.markdown("---")

    # Model resolution
    model_path, is_fallback = find_model_path()

    if is_fallback:
        st.warning(
            "⚠️ **Custom weights (`best.pt`) not found** in `model/` or `models/`.\n"
            "Using fallback pretrained: `yolo11n.pt`."
        )
    else:
        st.success(f"🎯 **Model Loaded:** `{os.path.basename(os.path.dirname(model_path))}/{os.path.basename(model_path)}`")

    st.markdown("### 🎚️ Detection Thresholds")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.25,
        step=0.05,
        help="Filters detections with a confidence score below this threshold.",
    )

    iou_threshold = st.slider(
        "IoU Threshold (NMS)",
        min_value=0.10,
        max_value=1.00,
        value=0.45,
        step=0.05,
        help="Intersection-Over-Union threshold for Non-Maximum Suppression.",
    )

    st.markdown("---")
    st.markdown("### 🔬 Pre-processing (CLAHE)")
    st.info(
        "**Color Space:** LAB (Equalized L channel)\n\n"
        "**Clip Limit:** `3.0`\n\n"
        "**Tile Grid:** `(8, 8)`"
    )

    st.markdown("---")
    st.markdown("### 📜 Detection History")
    history_df = get_detection_history()
    st.write(f"Total Logged Runs: **{len(history_df)}**")

    if not history_df.empty:
        csv_data = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Export History (CSV)",
            data=csv_data,
            file_name="aqua_vision_detection_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        if st.button("🗑️ Clear History", use_container_width=True):
            clear_history()
            st.rerun()


# ---------------------------------------------------------
# Main Page Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="aqua-title-container">
        <div class="aqua-title">🌊 Aqua Vision: Underwater Marine Species Detector</div>
        <div class="aqua-subtitle">
            Enhance low-contrast underwater imagery via OpenCV CLAHE (LAB color space) and run real-time marine species detection with fine-tuned YOLO11.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load YOLO model
try:
    with st.spinner("Loading YOLO11 model weights..."):
        model = load_yolo_model(model_path)
except Exception as e:
    st.error(f"Error loading YOLO model: {e}")
    st.stop()


# ---------------------------------------------------------
# Image Input Section
# ---------------------------------------------------------
col_input, col_sample = st.columns([3, 1])

with col_input:
    uploaded_file = st.file_uploader(
        "Upload an underwater image (JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        help="Select an underwater photo containing marine species.",
    )

with col_sample:
    st.write("&nbsp;")
    st.write("&nbsp;")
    use_sample = st.checkbox(
        "Use Sample Underwater Image",
        value=False,
        help="Load the bundled sample marine life image for instant verification.",
    )

# Determine image source
pil_image: Optional[Image.Image] = None
image_filename = ""

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file).convert("RGB")
        image_filename = uploaded_file.name
    except Exception as err:
        st.error(f"Error loading uploaded image: {err}")
elif use_sample:
    sample_path = get_wallpaper_path()
    if sample_path and os.path.isfile(sample_path):
        try:
            pil_image = Image.open(sample_path).convert("RGB")
            image_filename = os.path.basename(sample_path)
        except Exception as err:
            st.error(f"Error opening sample image: {err}")
    else:
        st.warning("Sample image not found in Pictures folder.")

# ---------------------------------------------------------
# Processing Pipeline & Results
# ---------------------------------------------------------
if pil_image is not None:
    # 1. Apply CLAHE Enhancement
    with st.spinner("Applying OpenCV CLAHE Enhancement (LAB L-channel)..."):
        enhanced_rgb = apply_clahe(pil_image, clip_limit=3.0, tile_grid_size=(8, 8))

    # 2. Run Inference
    with st.spinner("Running YOLO11 object detection..."):
        inference_result = run_inference(
            model=model,
            image_input=enhanced_rgb,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
        )

    # 3. Log run into SQLite database (deduplicated per image + threshold change)
    run_key = f"{image_filename}_{conf_threshold}_{iou_threshold}_{inference_result['total_count']}"
    if st.session_state.get("last_logged_run") != run_key:
        summary_str = json.dumps(inference_result["class_counts"])
        log_detection(
            filename=image_filename,
            total_detected=inference_result["total_count"],
            highest_confidence=inference_result["highest_conf"],
            class_summary=summary_str,
        )
        st.session_state["last_logged_run"] = run_key

    # 4. Side-by-Side Comparison Layout
    st.markdown("---")
    st.subheader("🔍 Side-by-Side Comparison")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 🌊 Enhanced Image (CLAHE)")
        st.image(
            enhanced_rgb,
            use_container_width=True,
            caption="Enhanced Image: LAB L-channel CLAHE (clipLimit=3.0, grid=(8,8))",
        )
        with st.expander("Compare with Original Uploaded Image"):
            st.image(pil_image, use_container_width=True, caption="Original Input Image")

    with col_right:
        st.markdown("#### 🎯 YOLO11 Detections")
        st.image(
            inference_result["annotated_image"],
            use_container_width=True,
            caption=f"Annotated Detections ({inference_result['total_count']} objects identified)",
        )

    # 5. Analytics Dashboard
    st.markdown("---")
    st.subheader("📊 Marine Species Detection Analytics")

    # Overview Metrics Row
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(
            label="Total Objects Detected",
            value=inference_result["total_count"],
        )
    with m_col2:
        st.metric(
            label="Highest Confidence",
            value=f"{inference_result['highest_conf'] * 100:.1f}%" if inference_result["highest_conf"] > 0 else "0.0%",
        )
    with m_col3:
        unique_classes = len(inference_result["class_counts"])
        st.metric(
            label="Distinct Species",
            value=unique_classes,
        )

    # Per-class counts row
    if inference_result["class_counts"]:
        st.markdown("##### 🐠 Per-Class Species Counts")
        class_items = list(inference_result["class_counts"].items())
        # Display in rows of 4 columns
        num_cols = min(max(len(class_items), 1), 5)
        cols = st.columns(num_cols)
        for i, (cls_name, count) in enumerate(class_items):
            with cols[i % num_cols]:
                st.metric(label=cls_name, value=count)
    else:
        st.info("No marine species detected above the chosen confidence threshold. Try lowering the threshold in the sidebar.")

    # Raw Detection Table (Filterable)
    st.markdown("---")
    st.subheader("📋 Raw Detection Details")

    detections_df: pd.DataFrame = inference_result["detections_df"]

    if not detections_df.empty:
        # Filter controls
        f_col1, f_col2 = st.columns([2, 2])
        all_classes = sorted(detections_df["Class"].unique().tolist())
        with f_col1:
            selected_classes = st.multiselect(
                "Filter by Species Class:",
                options=all_classes,
                default=all_classes,
            )
        with f_col2:
            min_filter_conf = st.slider(
                "Filter by Minimum Confidence:",
                min_value=float(conf_threshold),
                max_value=1.0,
                value=float(conf_threshold),
                step=0.01,
            )

        # Apply filters
        display_df = detections_df[
            (detections_df["Class"].isin(selected_classes))
            & (detections_df["Confidence Score"] >= min_filter_conf)
        ]

        # Table presentation
        st.dataframe(
            display_df[["Class", "Confidence Score", "Bounding Box (x1, y1, x2, y2)"]],
            use_container_width=True,
            hide_index=True,
        )

        # Download Table Button
        csv_download = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Current Detection Table (CSV)",
            data=csv_download,
            file_name=f"{image_filename}_detections.csv",
            mime="text/csv",
        )
    else:
        st.write("No raw detections to display.")

else:
    # Instructions and placeholder when no image is loaded
    st.info(
        "👋 **Welcome to Aqua Vision!**\n\n"
        "1. Upload an underwater marine photograph or check **'Use Sample Underwater Image'** above.\n"
        "2. The system will automatically enhance contrast using OpenCV CLAHE in LAB color space.\n"
        "3. YOLO11 will perform real-time species detection with bounding boxes and confidence scores.\n"
        "4. View the side-by-side comparison, analytics cards, and filterable raw detections below."
    )
