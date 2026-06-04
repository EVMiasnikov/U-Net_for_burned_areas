# UNet Burned Area Mapping

Semantic segmentation of burned/not-burned areas using UNet,
trained on PlanetScope imagery with optional Sentinel-2 and Landsat enhancement.

## Overview
This project provides a complete pipeline for:
- Training UNet on PlanetScope imagery
- Fine-tuning with Sentinel-2 and/or Landsat data
- Predicting burned areas on new imagery

## Project Structure
project_root/                        ← main folder (on Google Drive)
├── data/
│   ├── sites/
│   │   └── {site_name}/
│   │       ├── raw/                 ← downloaded imagery
│   │       └── patches/
│   │           └── {patches_folder}/
│   └── merged_patches_datasets/
│       └── {dataset_name}/
├── models/
│   └── final/
│       └── {experiment}.pt
├── predictions/
├── notebooks/
│   ├── load_sentinel_landsat.ipynb
│   ├── patch_creation.ipynb
│   ├── merging_patches.ipynb
│   ├── training.ipynb
│   └── prediction.ipynb
└── README.md

## Quick Start
1. Download this project folder to your Google Drive
2. Follow the pipeline described in the [Wiki](link-to-wiki)

## Requirements
- Google Colab
- Python 3.10+
