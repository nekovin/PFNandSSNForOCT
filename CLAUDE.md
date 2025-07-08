# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository implements FPSS (Fully Paired Speckle Separation), a deep learning approach for OCT (Optical Coherence Tomography) image denoising. The codebase has been refactored to follow professional software engineering practices with a unified framework for training, evaluation, and model management.

## Quick Start

### Installation
```bash
pip install -e .
```

### Training with Unified Framework (Recommended)
```bash
# Set dataset path
export DATASET_DIR_PATH="/path/to/OCT/dataset"

# Train with default configuration
python scripts/train_fpss_unified.py

# Train with custom configuration
python scripts/train_fpss_unified.py --config configs/fpss_default.yaml

# Train small model for quick testing
python scripts/train_fpss_unified.py --config configs/fpss_small.yaml
```

### Training with Command Line Options
```bash
# Override specific parameters
python scripts/train_fpss_unified.py \
    --model fpss_attention \
    --epochs 50 \
    --batch-size 8 \
    --learning-rate 0.002 \
    --dataset-root /path/to/dataset
```

## Unified Framework Architecture

The codebase has been restructured into a professional framework with clear separation of concerns:

### Core Components

1. **Model Factory** (`src/fpss/models/model_factory.py`)
   - Centralized model creation and management
   - Consistent interfaces for all model architectures
   - Support for model loading/saving and configuration

2. **Unified Loss Functions** (`src/fpss/losses/unified_losses.py`)
   - All loss functions consolidated into single module
   - Factory pattern for loss function creation
   - Custom FPSS loss with multiple components

3. **Unified Data Loading** (`src/fpss/data/unified_dataloader.py`)
   - Abstract base classes for consistent data interfaces
   - Support for SD-OCT and paired OCT datasets
   - Standardized preprocessing pipelines

4. **Unified Training Framework** (`src/fpss/training/unified_trainer.py`)
   - Configuration-driven training with YAML support
   - Built-in logging, checkpointing, and visualization
   - Mixed precision training and early stopping

5. **Shared Components** (`src/fpss/models/components/shared_components.py`)
   - Reusable neural network components
   - CBAM attention mechanisms
   - U-Net building blocks with consistent interfaces

### Available Models

- `fpss_attention`: Main FPSS model with attention (32 features, depth 4)
- `fpss_attention_small`: Lightweight variant (16 features, depth 3)
- `fpss_no_attention`: Baseline without attention (32 features, depth 5)

### Configuration System

Training is controlled via YAML configuration files in `configs/`:

- `fpss_default.yaml`: Standard configuration for full training
- `fpss_small.yaml`: Lightweight configuration for testing

Key configuration sections:
- Model architecture parameters
- Dataset and data loading settings
- Training hyperparameters
- Loss function configuration
- Logging and checkpointing options

### Dataset Structure
```
DATASET_DIR_PATH/
├── 0/  # Non-diabetic patients
├── 1/  # Diabetic type 1 patients
└── 2/  # Diabetic type 2 patients
    └── RawDataQA-* folders containing OCT scan sequences
```

## Professional Features

### Metrics and Evaluation
- Comprehensive metrics calculation (PSNR, SSIM, edge preservation, vessel continuity)
- Automated visualization of training progress and results
- Detailed evaluation reports with multiple image quality metrics

### Code Organization
- Type hints throughout the codebase
- Comprehensive docstrings following Google style
- Abstract base classes for extensibility
- Factory patterns for component creation
- Proper error handling and logging

### Training Features
- Mixed precision training for faster GPU utilization
- Automatic checkpointing with best model saving
- Early stopping with configurable patience
- TensorBoard logging for training visualization
- Configurable learning rate scheduling

## Development Workflow

1. **Setup**: Install dependencies and set `DATASET_DIR_PATH`
2. **Configuration**: Create or modify YAML config files
3. **Training**: Use `train_fpss_unified.py` with appropriate config
4. **Monitoring**: Check TensorBoard logs and saved visualizations
5. **Evaluation**: Models automatically save metrics and generate reports

## Legacy Code Note

The original training scripts in `scripts/fpss/`, `scripts/n2n/`, etc. are preserved for compatibility but the unified framework is recommended for new development.

## Important Notes

- All training outputs (checkpoints, logs, visualizations) are organized in separate directories
- CUDA is recommended for training with automatic device detection
- Mixed precision training is enabled by default for better performance
- Configuration files support environment variable substitution