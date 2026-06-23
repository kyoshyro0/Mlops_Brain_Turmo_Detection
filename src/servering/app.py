"""
Streamlit frontend for Brain Tumor Detection.

Communicates with the FastAPI backend to run inference and display results.
Provides a sidebar toggle to switch between PyTorch and ONNX models.
"""

import streamlit as st
import os
import requests
from PIL import Image, ImageDraw
import io
import hashlib
import time

# ── Configuration ──────────────────────────────────────────────────────────────

API_URL = os.environ.get("API_URL", "http://localhost:8000")
MAX_CACHE_ENTRIES = 20
MAX_IMAGE_SIZE = 640


# ── UI Components ──────────────────────────────────────────────────────────────

def setup_page():
    """Configure page settings."""
    st.set_page_config(
        page_title="Brain Tumor Detection",
        page_icon="🧠",
        layout="wide",
    )
    st.title("🧠 Brain Tumor Detection")
    st.markdown("Upload an MRI image to detect brain tumors using YOLOv11s")


def render_sidebar():
    """Render sidebar with model settings. Returns (confidence, use_onnx)."""
    st.sidebar.title("⚙️ Settings")

    confidence = st.sidebar.slider(
        "Confidence Threshold", 0.0, 1.0, 0.5, 0.05,
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Model Backend")
    use_onnx = st.sidebar.toggle(
        "Use ONNX (faster CPU inference)",
        value=False,
        help="Toggle to switch between PyTorch (GPU-optimised) and ONNX (CPU-optimised) backends.",
    )

    # Show current model info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Model Status")
    try:
        resp = requests.get(f"{API_URL}/model-info", timeout=3)
        if resp.status_code == 200:
            info = resp.json()
            active_backend = "ONNX" if use_onnx and info.get("onnx_available") else "PyTorch"
            st.sidebar.info(f"**Source:** {info.get('model_source', 'Unknown')}")
            st.sidebar.info(f"**Active backend:** {active_backend}")
            if not info.get("onnx_available") and use_onnx:
                st.sidebar.warning("⚠️ ONNX not available, will fallback to PyTorch")
        else:
            st.sidebar.warning("Cannot fetch model info")
    except requests.exceptions.RequestException:
        st.sidebar.error("API not reachable")

    return confidence, use_onnx


def check_api_status():
    """Check API connectivity."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            gpu = data.get("gpu_name", "CPU only")
            st.success(f"✅ API running — GPU: {gpu}")
        else:
            st.error("❌ API not responding correctly")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the API server is running.")
    except requests.exceptions.Timeout:
        st.warning("⏳ API is starting up (loading model). Please refresh in a few seconds.")


# ── Logic & Processing ─────────────────────────────────────────────────────────

def resize_image(image, max_size=MAX_IMAGE_SIZE):
    """Resize image maintaining aspect ratio."""
    orig_width, orig_height = image.size
    if max(orig_width, orig_height) > max_size:
        if orig_width > orig_height:
            new_width = max_size
            new_height = int(orig_height * (max_size / orig_width))
        else:
            new_height = max_size
            new_width = int(orig_width * (max_size / orig_height))

        image_resized = image.resize((new_width, new_height), Image.LANCZOS)
        scale_x = orig_width / new_width
        scale_y = orig_height / new_height
        return image_resized, scale_x, scale_y
    return image, 1.0, 1.0


def get_cache_key(file_bytes, confidence, use_onnx):
    """Generate unique cache key."""
    settings_str = f"{confidence}_{use_onnx}"
    return hashlib.md5(file_bytes + settings_str.encode()).hexdigest()


def make_prediction(image_bytes, use_onnx, confidence):
    """Send prediction request to API."""
    files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
    params = {"use_onnx": use_onnx, "confidence": confidence}

    try:
        response = requests.post(
            f"{API_URL}/predict", files=files, params=params, timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def draw_results(image, predictions, scale_x, scale_y, confidence_threshold):
    """Draw bounding boxes on image."""
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)

    colors = {
        "glioma": (255, 0, 0),
        "meningioma": (0, 255, 0),
        "pituitary": (0, 0, 255),
        "notumor": (255, 255, 0),
    }

    for pred in predictions:
        if pred["confidence"] >= confidence_threshold:
            bbox = pred["bbox"]
            if isinstance(bbox, list) and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = [
                    int(c * s)
                    for c, s in zip(
                        [x1, y1, x2, y2],
                        [scale_x, scale_y, scale_x, scale_y],
                    )
                ]
                color = colors.get(pred["class"], (255, 0, 0))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                label = f"{pred['class']}: {pred['confidence']:.2f}"
                draw.text((x1, y1 - 15), label, fill=color)

    return result_image


# ── Main Application ───────────────────────────────────────────────────────────

def main():
    setup_page()

    # Session state
    if "prediction_cache" not in st.session_state:
        st.session_state.prediction_cache = {}

    confidence, use_onnx = render_sidebar()
    check_api_status()

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"],
    )

    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        cache_key = get_cache_key(file_bytes, confidence, use_onnx)

        image = Image.open(uploaded_file)
        image_resized, scale_x, scale_y = resize_image(image)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # Cache check
        if cache_key in st.session_state.prediction_cache:
            result = st.session_state.prediction_cache[cache_key]
            is_cached = True
        else:
            is_cached = False
            with st.spinner("Detecting tumors..."):
                img_byte_arr = io.BytesIO()
                image_resized.convert("RGB").save(img_byte_arr, format="JPEG")

                start_time = time.time()
                result = make_prediction(
                    img_byte_arr.getvalue(), use_onnx, confidence,
                )
                processing_time = time.time() - start_time

                if "error" not in result:
                    result["client_processing_time"] = processing_time
                    if len(st.session_state.prediction_cache) >= MAX_CACHE_ENTRIES:
                        oldest = next(iter(st.session_state.prediction_cache))
                        st.session_state.prediction_cache.pop(oldest)
                    st.session_state.prediction_cache[cache_key] = result

        # Display results
        predictions = result.get("predictions", [])
        error = result.get("error")

        with col2:
            st.subheader("Detection Results")
            if error:
                st.error(f"Error: {error}")
            elif predictions:
                result_img = draw_results(
                    image, predictions, scale_x, scale_y, confidence,
                )
                st.image(
                    result_img, caption="Detection Results",
                    use_container_width=True,
                )
            else:
                st.info("No tumors detected")
                st.image(
                    image, caption="No tumors detected",
                    use_container_width=True,
                )

        # Info section
        st.subheader("Detection Information")
        info_cols = st.columns(3)
        with info_cols[0]:
            if is_cached:
                st.success("✅ Retrieved from cache")
            elif not error:
                st.metric("Processing Time", f"{result.get('client_processing_time', 0):.3f}s")
        with info_cols[1]:
            if not error:
                st.metric("Model", result.get("model_type", "Unknown"))
        with info_cols[2]:
            if not error:
                backend = result.get("model_framework", "Unknown")
                st.metric("Backend", backend)

        # Detection details
        if predictions:
            st.subheader("Detection Details")
            for p in predictions:
                if p["confidence"] >= confidence:
                    st.write(f"**{p['class']}**: {p['confidence']:.2f}")


if __name__ == "__main__":
    main()
