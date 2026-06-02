# Brain Tumor Segmentation Demo

This repository contains a Streamlit demo prepared for the brain tumor segmentation project.

The main study compares three segmentation approaches on BraTS2020 H5 brain MRI slices:

- U-Net
- SAM
- MedSAM-Optimized

The purpose of this demo is to provide a simple interactive interface where a user can upload a BraTS2020 `.h5` slice file and visualize the MRI channels, ground-truth tumor mask, overlay result, and model comparison outputs.

## Live Demo

Streamlit app:

https://deeplearningsegmentationgit-czo945jackpv2pvriqvdmb.streamlit.app/

## Project Purpose

This project is based on the proposal titled:

**A Comparative Study of Task-Specific and Foundation Models for Brain Tumor Segmentation**

The study investigates how a task-specific deep learning model, U-Net, compares with foundation-based segmentation models, SAM and MedSAM, for brain tumor segmentation.

The final notebook results showed the following performance:

| Model | Dice | IoU | Description |
|---|---:|---:|---|
| U-Net | 0.7583 | 0.6720 | Supervised model trained on BraTS2020 |
| MedSAM-Optimized | 0.7338 | 0.6143 | Medical domain-adapted foundation model |
| SAM | 0.6304 | 0.4992 | General-purpose segmentation foundation model |

The results show that U-Net achieved the highest overall performance because it was trained directly on the BraTS2020 dataset. MedSAM-Optimized performed better than SAM, which supports the importance of medical domain adaptation for foundation models. Therefore, the initial hypothesis was partially supported: MedSAM improved over SAM, but it did not outperform the supervised U-Net baseline.

## Demo Description

The instructor requested a demo for the project. Therefore, this Streamlit application was created to demonstrate the workflow in an interactive way.

The demo allows the user to:

- Upload a BraTS2020 `.h5` slice file
- Display the H5 file information
- Visualize the four MRI image channels
- Show the binary whole tumor ground-truth mask
- Display tumor overlay visualization
- Compare U-Net, SAM, and MedSAM-Optimized outputs
- View the final notebook performance results

## Sample H5 Files

For testing the demo, five sample BraTS2020 H5 slice files are included in this repository.

Example files:

- `volume_100_slice_100.h5`
- `volume_293_slice_90.h5`
- `volume_328_slice_88.h5`
- `volume_328_slice_89.h5`
- `volume_328_slice_91.h5`

These files can be uploaded directly into the Streamlit demo using the upload button.

Each H5 file contains:

```text
image → shape: (240, 240, 4)
mask  → shape: (240, 240, 3)
