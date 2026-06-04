# Project Name

Brief description of what this project does.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
...

# UNet Patch Pipeline

Short description — what, for what task, on what data.

## Project Structure
├── notebooks/
│   ├── patch_creation.ipynb
│   └── patch_merging.ipynb
├── patches/
│   └── v1_size256_overlap32/
└── README.md

## Requirements
- Python 3.10+
- torch, torchvision, numpy...

## Usage

### 1. Patch Creation (per site)
Open `patch_creation.ipynb` and set parameters:
- `SITE_NAME` = "site_A"
- `PATCH_SIZE` = 256

### 2. Patch Merging & Training
Open `patch_merging.ipynb`, specify sites to merge.

## Parameters
| Parameter   | Default | Description          |
|-------------|---------|----------------------|
| patch_size  | 256     | Size of each patch   |
| overlap     | 32      | Overlap between patches |

## Versioning
Patches are stored with version tags: `v{n}_size{s}_overlap{o}/`
