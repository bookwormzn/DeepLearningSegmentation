import streamlit as st
import numpy as np
import h5py
import pandas as pd
import cv2

st.set_page_config(
    page_title="Brain Tumor Segmentation Demo",
    layout="wide"
)

st.title("Brain Tumor Segmentation Demo")
st.write(
    "A Streamlit demo based on the final notebook: "
    "U-Net, SAM, and MedSAM comparison for BraTS2020 H5 brain tumor slices."
)

st.sidebar.header("Demo Settings")

model_choice = st.sidebar.selectbox(
    "Choose model",
    ["U-Net", "SAM", "MedSAM-Optimized", "Compare All"]
)

show_overlay = st.sidebar.checkbox("Show overlay", value=True)

st.sidebar.markdown("---")
st.sidebar.write("Notebook final test results")

results_df = pd.DataFrame({
    "Model": ["U-Net", "MedSAM-Optimized", "SAM"],
    "Dice": [0.7583, 0.7338, 0.6304],
    "IoU": [0.6720, 0.6143, 0.4992],
    "Training Type": [
        "Supervised on BraTS2020",
        "Zero-shot medical foundation model",
        "Zero-shot general foundation model"
    ]
})

st.sidebar.dataframe(results_df, use_container_width=True)

uploaded_h5 = st.file_uploader(
    "Upload a BraTS2020 H5 slice file",
    type=["h5", "hdf5"]
)


def normalize_to_uint8(image):
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.uint8)

    image = (image - min_val) / (max_val - min_val)
    image = (image * 255).astype(np.uint8)
    return image


def make_binary_tumor_mask(mask):
    """
    Converts the BraTS mask into one binary Whole Tumor mask.
    """
    if mask.ndim == 3:
        binary_mask = np.any(mask > 0, axis=-1).astype(np.uint8)
    else:
        binary_mask = (mask > 0).astype(np.uint8)

    return binary_mask


def create_rgb_from_channels(image, channels=(0, 1, 2)):
    """
    Converts 4-channel BraTS MRI image into 3-channel RGB-like image.
    """
    if image.ndim == 2:
        img = normalize_to_uint8(image)
        return np.stack([img, img, img], axis=-1)

    selected = []
    for ch in channels:
        selected.append(normalize_to_uint8(image[:, :, ch]))

    rgb = np.stack(selected, axis=-1)
    return rgb


def create_overlay(image_rgb, mask, color=(255, 0, 0), alpha=0.45):
    """
    Overlays tumor mask on MRI image.
    """
    overlay = image_rgb.copy()

    color_mask = np.zeros_like(image_rgb)
    color_mask[mask > 0] = color

    overlay = cv2.addWeighted(overlay, 1.0, color_mask, alpha, 0)
    return overlay


def dummy_prediction_from_gt(mask, model_name):
    """
    Temporary demo prediction.

    This is NOT the real model prediction.
    It only creates visual masks for Streamlit demo testing.
    """
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

    if denominator == 0:
        return 1.0

    return 2.0 * intersection / denominator


def iou_score(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    if union == 0:
        return 1.0

    return intersection / union


if uploaded_h5 is None:
    st.info("Please upload a BraTS2020 .h5 file to start.")

    st.subheader("Final Notebook Results")
    st.dataframe(results_df, use_container_width=True)

    st.markdown(
        """
        **Interpretation:**  
        The final notebook results show that U-Net achieved the highest Dice and IoU scores, 
        followed by MedSAM-Optimized and SAM. This means that supervised U-Net training is 
        still very strong for BraTS2020, while MedSAM improves clearly over general-purpose SAM.
        """
    )

else:
    with h5py.File(uploaded_h5, "r") as f:
        keys = list(f.keys())

        if "image" not in keys or "mask" not in keys:
            st.error("This H5 file must contain 'image' and 'mask' keys.")
            st.stop()

        image = np.array(f["image"])
        mask = np.array(f["mask"])

    st.subheader("H5 File Information")

    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        st.metric("Image shape", str(image.shape))

    with col_info2:
        st.metric("Mask shape", str(mask.shape))

    with col_info3:
        st.metric("Available keys", ", ".join(keys))

    binary_mask = make_binary_tumor_mask(mask)

    st.subheader("MRI Modalities / Channels")

    if image.ndim == 3:
        cols = st.columns(image.shape[-1])

        for ch in range(image.shape[-1]):
            channel_img = normalize_to_uint8(image[:, :, ch])
            with cols[ch]:
                st.image(
                    channel_img,
                    caption=f"Channel {ch}",
                    use_container_width=True,
                    clamp=True
                )
    else:
        st.image(
            normalize_to_uint8(image),
            caption="MRI Image",
            use_container_width=True,
            clamp=True
        )

    st.subheader("Ground Truth Tumor Mask")

    rgb_image = create_rgb_from_channels(image, channels=(0, 1, 2))

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            rgb_image,
            caption="RGB-like MRI Input from Channels (0, 1, 2)",
            use_container_width=True
        )

    with col2:
        st.image(
            binary_mask * 255,
            caption="Binary Whole Tumor Mask",
            use_container_width=True,
            clamp=True
        )

    if show_overlay:
        st.subheader("Ground Truth Overlay")
        gt_overlay = create_overlay(rgb_image, binary_mask, color=(255, 0, 0), alpha=0.45)
        st.image(
            gt_overlay,
            caption="Tumor Mask Overlay on MRI",
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Segmentation Demo")

    st.warning(
        "This Streamlit version currently uses demo predictions for interface testing. "
        "Real U-Net, SAM and MedSAM inference can be connected later using model checkpoints."
    )

    if st.button("Run Segmentation Demo"):
        if model_choice == "Compare All":
            models = ["U-Net", "SAM", "MedSAM-Optimized"]
        else:
            models = [model_choice]

        output_rows = []
        cols = st.columns(len(models))

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
                pred_overlay = create_overlay(rgb_image, pred, color=(0, 255, 0), alpha=0.45)
                display_img = pred_overlay
                caption = f"{model_name} Prediction Overlay"
            else:
                display_img = pred * 255
                caption = f"{model_name} Prediction Mask"

            with cols[idx]:
                st.image(
                    display_img,
                    caption=caption,
                    use_container_width=True,
                    clamp=True
                )

        st.subheader("Demo Metrics for Uploaded Slice")
        st.dataframe(pd.DataFrame(output_rows), use_container_width=True)

    st.markdown("---")

    st.subheader("Final Notebook Model Comparison")
    st.dataframe(results_df, use_container_width=True)

    st.markdown(
        """
        **Notebook-based interpretation:**  
        U-Net achieved the best overall performance because it was trained directly on BraTS2020.  
        MedSAM-Optimized performed better than SAM, showing the benefit of medical domain adaptation.  
        SAM had the lowest performance because it is a general-purpose segmentation model and was not adapted to MRI tumor segmentation.
        """
    )
