# UNet Burned Area Mapping

Semantic segmentation of burned/not-burned areas using UNet,
trained on PlanetScope imagery with optional Sentinel-2 and Landsat enhancement.

## Overview
This project provides a complete pipeline for:
- Training UNet on PlanetScope imagery and ground trough polygons
- Fine-tuning with Sentinel-2 and/or Landsat data
- Predicting burned areas on new imagery

## Project Structure
```
project_root/                        ← main folder (on Google Drive)
├── data/
│   ├── sites/
│   │   └── {site_name}/
│   │       ├── 385_predictors.tif                ← Planetscope predictors
│   │       ├── 385_polygons.gpkg                 ← Ground truth
│   │       ├── patches/
│   │       │   └── {patches_folder}/
│   │       │       ├── test/
│   │       │       ├── train/
│   │       │       ├── val/
│   │       │       └── metadata.json
│   │       ├── Sentinel
│   │       └── Landsat
│   └── merged_patches_datasets/
│       └── {dataset_name}/
│           ├── test/
│           ├── train/
│           ├── val/
│           └── metadata.json
├── experiments/
│   └── {experiment}/
├── models/
│   ├── final/
│   │   └── {experiment}.pt
│   └── checkpoints/
│       └── {experiment}/
│           └── {experiment_epoch_n}.pt
├── Predictions/
├── notebooks/
│   ├── User_Satellite_Downloads.ipynb
│   ├── Patch_creation.ipynb
│   ├── Merging_patches.ipynb
│   ├── Training.ipynb
│   ├── Fine-tuning.ipynb
│   └── Prediction.ipynb
├── src/
├── Pre+Post_Downloads/
├── User's_Inputs/
│   ├── AOI_Polygons/
│   └── Input_Rasters_For_Prediction/
├── tesnorboard/
└── README.md
```
## Quick Start
1. Download and unzip this project folder to your Google Drive
2. Follow the pipeline described in the [Wiki](https://github.com/EVMiasnikov/U-Net_for_burned_areas/wiki)

## Requirements
- Google Colab
- Python 3.10+
