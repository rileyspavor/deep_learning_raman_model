# V3 Data Training Guide

This guide explains how to train the CNN model on the v3 data without preprocessing.

## Overview

The `train_v3.py` script loads Raman spectra from the v3 data folder and trains a CNN model directly without any preprocessing steps (no alignment, baseline correction, normalization, or smoothing). The data is assumed to be already prepared and ready for training.

## Dataset Information

Based on the methodology document, the v3 dataset contains:
- **32,000 total spectra**
- **9 classes** (graphene-related materials)
- **Balanced dataset** (4000 spectra per class)
- **70/15/15 train/val/test split** (may already be in the npz file)
- **Int16 format** (automatically converted to float32 for PyTorch)

## Usage

### Basic Usage

Train with default settings:
```bash
python train_v3.py --data "v3 data/synthetic_graphene_parametric_9class_v2.npz"
```

### With Custom Settings

```bash
python train_v3.py \
    --data "v3 data/synthetic_graphene_parametric_9class_v2.npz" \
    --output "saved_models_v3" \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001
```

### Using Existing Splits

If the npz file already contains train/val/test splits:
```bash
python train_v3.py \
    --data "v3 data/synthetic_graphene_parametric_9class_v2.npz" \
    --use-existing-splits
```

## Command Line Arguments

- `--data`: Path to the npz file (default: `"v3 data/synthetic_graphene_parametric_9class_v2.npz"`)
- `--output`: Output directory for saved models (default: `"saved_models_v3"`)
- `--epochs`: Number of training epochs (default: `100`)
- `--batch-size`: Batch size for training (default: `32`)
- `--lr`: Learning rate (default: `0.001`)
- `--use-existing-splits`: Use existing train/val/test splits from metadata if available

## Expected NPZ File Structure

The npz file should contain:
- `spectra`: Array of shape `(n_samples, n_wavenumbers)` - Raman spectra
- `wavenumbers`: Array of shape `(n_wavenumbers,)` - common wavenumber grid
- `y`: Array of shape `(n_samples,)` - integer class labels (0 to n_classes-1)
- `label_names`: (Optional) Array of class name strings

Additional metadata keys (like `id_ig`, `i2d_ig`) will be loaded but not used for training.

## Output Files

After training, the following files will be saved in the output directory:

- `model_checkpoint_v3.pth`: Full model checkpoint with optimizer state
- `model_state_v3.pth`: Model weights only
- `target_grid_v3.npy`: Wavenumber grid
- `class_names_v3.json`: Class name mapping
- `training_history_v3.json`: Training history (loss and accuracy over epochs)
- `test_metrics_v3.json`: Detailed test set metrics
- `training_history_v3.png`: Plot of training/validation loss and accuracy
- `confusion_matrix_v3.png`: Confusion matrix for test set

## Model Architecture

The model uses a 1D CNN architecture:
- **Convolutional layers**: [32, 64, 128, 256] channels
- **Kernel sizes**: [7, 5, 5, 3]
- **Pooling**: MaxPool1d with size 2 after each conv layer
- **Fully connected**: [128, 64] hidden units
- **Dropout**: 0.3
- **Batch normalization**: Enabled

## Training Details

- **Loss function**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001, weight_decay=1e-4)
- **Learning rate scheduler**: ReduceLROnPlateau (reduces LR when validation loss plateaus)
- **Early stopping**: Stops training if validation loss doesn't improve for 10 epochs
- **Data splits**: 70% train, 15% validation, 15% test (stratified)

## Notes

1. **No Preprocessing**: The script loads data directly without any preprocessing (no baseline correction, normalization, smoothing, etc.)

2. **Type Conversion**: Int16 data is automatically converted to float32 for PyTorch compatibility (this is just a type conversion, not preprocessing)

3. **Automatic Class Detection**: The number of classes is automatically detected from the labels in the dataset

4. **Stratified Splits**: If splits need to be created, they use stratified splitting to maintain class balance

## Example Output

```
============================================================
Loading V3 Data
============================================================

Dataset loaded successfully!
  Spectra shape: (32000, 2401)
  Wavenumbers shape: (2401,)
  Labels shape: (32000,)
  Spectra dtype: int16
  Converting spectra from int16 to float32 (type conversion only, no preprocessing)...
  Number of classes: 9
  ...
```



