# MIL-Based Amyloidosis Classification

[![DOI](https://zenodo.org/badge/770444738.svg)](https://doi.org/10.5281/zenodo.15482070)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/IMIS-MIKI/llm-exams-evaluation/blob/main/LICENSE)

This repository contains code and materials associated with the paper:

**Enhancing Amyloidosis Classification through Multiple Instance Learning: A Histopathological Image Analysis**

## Overview

This project presents a deep learning pipeline using **Multiple Instance Learning (MIL)** to classify cardiac amyloidosis into AL and ATTR subtypes. The approach leverages fluorescent histopathological images of Congo red-stained heart tissue and focuses on morphological features such as amyloid deposit patterns.

Key results:
- **88.5% accuracy** on a blind test cohort
- Classification without the need for immunohistochemistry or mass spectrometry
- Application of MIL to distinguish structural differences between AL and ATTR amyloid deposits

## Features

- Full pipeline: from NDPI image preprocessing to tile extraction and MIL classification
- MIL architecture using a modified **ResNet34** backbone
- Advanced preprocessing: background removal, intensity normalization, tissue detection
- Bag-level modeling using [mil_pytorch](https://github.com/AmrElsersy/Multiple-Instance-Learning-Pytorch)
- Hyperparameter optimization included (tile size, tile count, background thresholds, aggregation methods)

## Repository Structure

```
├── data/                   # Input images and masks (not publicly available)
├── mil_pytorch/            # Edited Mil pytorch library
├── Model Storage/          # Storage of run models
├── test/                   # Scripts used to test different preprocessing approaches
├── utlis/                  # Auxiliary functions
├── main*                   # Main files used to train all data, train using CV-folds, test results...
├── README.md               # This file
├── requirements.txt        # Libraries required to run the project
```

## Requirements

- Python 3.8+

Install dependencies via:

```bash
pip install -r requirements.txt
```

## Ethics

All patient data was anonymized, and ethical approvals were obtained from:
- Kiel University Hospital Ethics Committee (D581-585/15; D469/18)
- Medical Faculty of the University of Heidelberg (S-093/2014)

## Data Availability

Due to patient confidentiality and ethical restrictions, **raw images are not publicly available**. However, code and synthetic examples may be released to replicate the pipeline.

## Citation

If you use this work, please cite:

**Macedo M. et al.** Enhancing Amyloidosis Classification through Multiple Instance Learning: A Histopathological Image Analysis, 2025.
