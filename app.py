import streamlit as st
import numpy as np
import h5py
import pandas as pd
import cv2

st.set_page_config(
    page_title="Brain Tumor Segmentation Demo",
    page_icon="🧠",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 46px;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        color: #CBD5E1;
        margin-bottom: 28px;
    }

    .info-card {
        background-color: #111827;
        padding: 22px;
        border-radius: 18px;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        margin-bottom: 18px;
    }

    .metric-card {
        background: linear-gradient(135deg, #0F172A, #1E293B);
        padding: 18px;
        border-radius: 16px;
        border: 1px solid #334155;
        text-align: center;
    }

    .metric-title {
        color: #94A3B8;
        font-size: 14px;
        margin-bottom: 6px;
    }

    .metric-value {
        color: #F8FAFC;
        font-size: 27px;
        font-weight: 700;
    }

    .section-title {
        font-size: 27px;
        font-weight: 750;
        margin-top: 28px;
        margin-bottom: 16px;
        color: #F8FAFC;
    }

    .small-note {
        color: #94A3B8;
        font-size: 14px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 26px;
    }

    div[data-testid="stFileUploader"] {
        background-color: #111827;
        padding: 18px;
        border-radius: 16px;
        border: 1px dashed #475569;
    }

    .stDataFrame {
        border-radius: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">🧠 Brain Tumor Segmentation Demo</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Interactive Streamlit demo for BraTS2020 H5 brain tumor slices using U-Net, SAM, and MedSAM comparison results.</div>',
    unsafe_allow_html=True
)

st.sidebar.title("🧪 Demo Settings")

model_choice = st.sidebar.selectbox(
    "Choose model",
    ["Compare All", "U-Net", "SAM", "MedSAM-Optimized"]
)

show_overlay = st.sidebar.checkbox("Show overlay", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Notebook Results")

results_df = pd.DataFrame({
    "Model": ["U-Net", "MedSAM-Optimized", "SAM"],
    "Dice": [0.7583, 0.7338, 0.6304],
    "IoU": [0.6720, 0.6143, 0.4992],
    "Training Type": [
        "Supervised",
        "Medical foundation",
        "General foundation"
    ]
})

st.sidebar.dataframe(
    results_df[["Model", "Dice", "IoU"]],
    use_container_width=True,
    hide_index=True
)

uploaded_h5 = st.file_uploader(
    "📤 Upload a BraTS2020 H5 slice file",
    type=["h5", "hdf5"]
)


def normalize_to_uint8(image):
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.uint8)

    image = (image - min_val) / (max_val - min_val)
    return (image * 255).astype(np.uint8)


def make_binary_tumor_mask(mask):
    if mask.ndim == 3:
        return np.any(mask > 0, axis=-1).astype(np.uint8)
    return (mask > 0).astype(np.uint8)


def create_rgb_from_channels(image, channels=(0, 1, 2)):
    if image.ndim == 2:
        img = normalize_to_uint8(image)
        return np.stack([img, img, img], axis=-1)

    selected = []
    for ch in channels:
        selected.append(normalize_to_uint8(image[:, :, ch]))

    return np.stack(selected, axis=-1)


def create_overlay(image_rgb, mask, color=(255, 64, 64), alpha=0.45):
    color_mask = np.zeros_like(image_rgb)
    color_mask[mask > 0] = color
    return cv2.addWeighted(image_rgb, 1.0, color_mask, alpha, 0)


def dummy_prediction_from_gt(mask, model_name):
    mask = mask.astype(np.uint8)

    if model_name == "U-Net":
        kernel = np.ones((3, 3), np.uint8)
        pred = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    elif model_name == "SAM":
        kernel = np.ones((9, 9), np.uint8)
        pred = cv2.dilate(mask, kernel, iterations=1)

    elif model_name == "MedSAM-Optimized":
        kernel = np.ones((5, 5), np.uint8)
        pred = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        pred = cv2.erode(pred, kernel, iterations=1)

    else:
        pred = mask.copy()

    return pred.astype(np.uint8)


def dice_score(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    intersection = np.logical_and(gt, pred).sum()
    denominator = gt.sum() + pred.sum()
    return 1.0 if denominator == 0 else 2.0 * intersection / denominator


def iou_score(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    return 1.0 if union == 0 else intersection / union


if uploaded_h5 is None:
    st.info("Upload one BraTS2020 `.h5` slice file to start the interactive demo.")

    st.markdown('<div class="section-title">📊 Final Notebook Results</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Best Overall Model</div>
                <div class="metric-value">U-Net</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">Best Dice Score</div>
                <div class="metric-value">0.7583</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-title">MedSAM vs SAM</div>
                <div class="metric-value">Improved</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="info-card">
        <b>Interpretation:</b><br>
        U-Net achieved the highest Dice and IoU because it was trained directly on BraTS2020.
        MedSAM-Optimized performed better than SAM, showing the benefit of medical domain adaptation.
        SAM had the lowest performance because it is a general-purpose segmentation model.
        </div>
        """,
        unsafe_allow_html=True
    )

else:
    with h5py.File(uploaded_h5, "r") as f:
        keys = list(f.keys())

        if "image" not in keys or "mask" not in keys:
            st.error("This H5 file must contain `image` and `mask` keys.")
            st.stop()

        image = np.array(f["image"])
        mask = np.array(f["mask"])

    binary_mask = make_binary_tumor_mask(mask)
    rgb_image = create_rgb_from_channels(image, channels=(0, 1, 2))

    st.markdown('<div class="section-title">📁 H5 File Summary</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric("Image Shape", str(image.shape))

    with m2:
        st.metric("Mask Shape", str(mask.shape))

    with m3:
        st.metric("Tumor Pixels", int(binary_mask.sum()))

    with m4:
        st.metric("Keys", ", ".join(keys))

    st.markdown('<div class="section-title">🧬 MRI Modalities</div>', unsafe_allow_html=True)

    channel_names = ["Channel 0", "Channel 1", "Channel 2", "Channel 3"]

    if image.ndim == 3:
        cols = st.columns(image.shape[-1])
        for ch in range(image.shape[-1]):
            with cols[ch]:
                st.image(
                    normalize_to_uint8(image[:, :, ch]),
                    caption=channel_names[ch] if ch < len(channel_names) else f"Channel {ch}",
                    use_container_width=True,
                    clamp=True
                )
    else:
        st.image(normalize_to_uint8(image), caption="MRI Image", use_container_width=True)

    st.markdown('<div class="section-title">🎯 Ground Truth Visualization</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(rgb_image, caption="RGB-like MRI Input", use_container_width=True)

    with col2:
        st.image(binary_mask * 255, caption="Binary Whole Tumor Mask", use_container_width=True)

    with col3:
        gt_overlay = create_overlay(rgb_image, binary_mask, color=(255, 64, 64), alpha=0.50)
        st.image(gt_overlay, caption="Ground Truth Overlay", use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">🤖 Segmentation Comparison</div>', unsafe_allow_html=True)

    st.warning(
        "This web demo uses visual demo masks for interface presentation. "
        "The real quantitative model evaluation was performed in the final notebook."
    )

    run_button = st.button("🚀 Run Segmentation Demo", use_container_width=True)

    if run_button:
        if model_choice == "Compare All":
            models = ["U-Net", "SAM", "MedSAM-Optimized"]
        else:
            models = [model_choice]

        output_rows = []
        model_cols = st.columns(len(models))

        for idx, model_name in enumerate(models):
            pred = dummy_prediction_from_gt(binary_mask, model_name)

            dice = dice_score(binary_mask, pred)
            iou = iou_score(binary_mask, pred)

            output_rows.append({
                "Model": model_name,
                "Demo Dice": round(dice, 4),
                "Demo IoU": round(iou, 4)
            })

            if show_overlay:
                pred_overlay = create_overlay(rgb_image, pred, color=(34, 197, 94), alpha=0.50)
                display_img = pred_overlay
                caption = f"{model_name} Overlay"
            else:
                display_img = pred * 255
                caption = f"{model_name} Mask"

            with model_cols[idx]:
                st.image(display_img, caption=caption, use_container_width=True, clamp=True)

        st.markdown('<div class="section-title">📈 Demo Metrics for Uploaded Slice</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(output_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📊 Final Notebook Model Comparison</div>', unsafe_allow_html=True)

    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="info-card">
        <b>Notebook-based conclusion:</b><br>
        The final notebook results show that U-Net achieved the best overall performance.
        MedSAM-Optimized was lower than U-Net but clearly better than SAM.
        Therefore, the hypothesis was partially supported: MedSAM improved over SAM,
        but did not outperform the supervised U-Net baseline.
        </div>
        """,
        unsafe_allow_html=True
    )
