import streamlit as st
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import cv2

st.set_page_config(
    page_title="Brain Tumor Segmentation Demo",
    layout="wide"
)

st.title("Brain Tumor Segmentation Demo")
st.write("BraTS H5 slice-based demo for U-Net, SAM and MedSAM.")

st.sidebar.header("Settings")

model_choice = st.sidebar.selectbox(
    "Choose model",
    ["U-Net", "SAM", "MedSAM", "Compare All"]
)

uploaded_h5 = st.file_uploader(
    "Upload a BraTS H5 file",
    type=["h5", "hdf5"]
)

def normalize_image(image):
    image = image.astype(np.float32)
    min_val = np.min(image)
    max_val = np.max(image)

    if max_val - min_val == 0:
        return image

    return (image - min_val) / (max_val - min_val)

def create_dummy_prediction(mask):
    """
    Temporary prediction.
    For now, we use the ground-truth mask shape only to test the interface.
    Later, this part will be replaced with U-Net, SAM and MedSAM predictions.
    """
    prediction = np.zeros_like(mask, dtype=np.uint8)

    if np.max(mask) > 0:
        prediction[mask > 0] = 1

    return prediction

if uploaded_h5 is not None:
    with h5py.File(uploaded_h5, "r") as f:
        st.subheader("H5 File Information")

        keys = list(f.keys())
        st.write("Available keys:", keys)

        image = np.array(f["image"])
        mask = np.array(f["mask"])

    st.write("Image shape:", image.shape)
    st.write("Mask shape:", mask.shape)

    if image.ndim == 3:
        channel_options = list(range(image.shape[-1]))
        selected_channel = st.sidebar.selectbox(
            "Select MRI channel",
            channel_options,
            index=0
        )
        image_2d = image[:, :, selected_channel]
    else:
        image_2d = image

    if mask.ndim == 3:
        mask_2d = mask[:, :, 0]
    else:
        mask_2d = mask

    image_2d = normalize_image(image_2d)
    mask_2d = (mask_2d > 0).astype(np.uint8)

    st.subheader("Input Slice and Ground Truth")

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            image_2d,
            caption="MRI Slice",
            use_container_width=True,
            clamp=True
        )

    with col2:
        st.image(
            mask_2d * 255,
            caption="Ground Truth Tumor Mask",
            use_container_width=True,
            clamp=True
        )

    if st.button("Run Segmentation"):
        with st.spinner("Running segmentation..."):
            prediction = create_dummy_prediction(mask_2d)

        st.subheader("Segmentation Result")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(
                image_2d,
                caption="MRI Slice",
                use_container_width=True,
                clamp=True
            )

        with col2:
            st.image(
                mask_2d * 255,
                caption="Ground Truth",
                use_container_width=True,
                clamp=True
            )

        with col3:
            st.image(
                prediction * 255,
                caption=f"{model_choice} Prediction",
                use_container_width=True,
                clamp=True
            )

        st.success("H5 file loaded and demo segmentation completed.")
else:
    st.info("Please upload a BraTS H5 file to start.")
