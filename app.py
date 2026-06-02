import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Brain Tumor Segmentation Demo",
    layout="wide"
)

st.title("Brain Tumor Segmentation Demo")
st.write("U-Net, SAM and MedSAM comparison demo.")

st.sidebar.header("Settings")

model_choice = st.sidebar.selectbox(
    "Choose model",
    ["U-Net", "SAM", "MedSAM", "Compare All"]
)

uploaded_image = st.file_uploader(
    "Upload an MRI image",
    type=["png", "jpg", "jpeg", "npy"]
)

uploaded_mask = st.file_uploader(
    "Optional: Upload ground-truth mask",
    type=["png", "jpg", "jpeg", "npy"]
)

def load_file(uploaded_file):
    if uploaded_file.name.endswith(".npy"):
        return np.load(uploaded_file)
    else:
        image = Image.open(uploaded_file).convert("L")
        return np.array(image)

def dummy_segmentation(image):
    """
    Temporary fake segmentation.
    We will replace this later with U-Net, SAM and MedSAM.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    center_h, center_w = h // 2, w // 2
    radius_h, radius_w = h // 6, w // 6

    mask[
        center_h - radius_h:center_h + radius_h,
        center_w - radius_w:center_w + radius_w
    ] = 255

    return mask

if uploaded_image is not None:
    image = load_file(uploaded_image)

    st.subheader("Uploaded MRI Image")
    st.image(image, caption="Input MRI Image", use_container_width=True)

    if uploaded_mask is not None:
        mask = load_file(uploaded_mask)
        st.subheader("Ground Truth Mask")
        st.image(mask, caption="Ground Truth Mask", use_container_width=True)

    if st.button("Run Segmentation"):
        with st.spinner("Running segmentation..."):
            prediction = dummy_segmentation(image)

        st.subheader("Predicted Tumor Mask")
        st.image(prediction, caption=f"{model_choice} Prediction", use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Selected Model", model_choice)

        with col2:
            st.metric("Status", "Demo prediction completed")
else:
    st.info("Please upload an MRI image to start.")
