# AD vs CN Classification for Vietnamese Dataset

## Overview

Alzheimer's Disease (AD) vs healthy control (CN) classification is framed as a binary classification objective for prediction of AD status, using single sequence MR input.

### Task Details

- Modelling: Binary Classification
- Input: Single T1-weighted MRI scan
- Output: Binary prediction (0: healthy control (CN), 1: Alzheimer's Disease (AD))
- Metric: AUC, F1-score

### Input Format
- MRI Format: NIFTI (.nii.gz)
- Image Size: 96×96×96 voxels (automatically resized)
- Sequences: T1-weighted (single sequence)

### CSV Format

Your CSV file should contain the following columns:

```csv
pat_id,label
subject001,0
subject002,1
subject003,0
subject004,1
```

### Directory Structure
Format the data structure as mentioned below
```
data/
├── images/
│   ├── subject001.nii.gz
│   ├── subject002.nii.gz
│   ├── subject003.nii.gz
│   └── subject004.nii.gz
└── csvs/
    ├── vn.csv
    ├── vn_train.csv
    ├── vn_train_k_1.csv
    ├── vn_train_k_5.csv
    ├── vn_val.csv
    └── vn_test.csv
```

## Configuration

Change the configurations in `src/config_vn_fewshot_k_1.yml`, `src/config_vn_fewshot_k_5.yml`, and `src/config_vn_linear_probing.yml`.

## Training

```bash
python src/train_lightning_mci.py --config /path/to/one/of/the/above/config
```

## Inference

### Run Inference
```bash
python src/test_inference_finetune.py
```