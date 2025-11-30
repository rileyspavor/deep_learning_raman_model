# .npz Format Support Guide

## Overview

The codebase now supports loading Raman spectroscopy data from `.npz` (NumPy compressed) files, which is convenient for storing pre-processed datasets.

## Expected .npz File Structure

Your `.npz` file should contain the following keys:

- **`spectra`**: Array of shape `(n_samples, n_wavenumbers)` - the Raman spectra
- **`wavenumbers`**: Array of shape `(n_wavenumbers,)` - common wavenumber grid (or `(n_samples, n_wavenumbers)` if different per spectrum)
- **`y`**: Array of shape `(n_samples,)` - integer class labels
- **`label_names`**: Array of strings - names for each class (optional)
- **Additional keys**: Any other data (e.g., `id_ig`, `i2d_ig`) will be stored as metadata

## Quick Start

### Option 1: Use the Example Script

```python
# Update the file path in example_npz_workflow.py
python example_npz_workflow.py
```

### Option 2: Load Manually

```python
import numpy as np
from data_ingestion import load_npz_dataset
from preprocessing import preprocess_aligned_spectra
from utils import stratified_split

# Load the .npz file
spectra, wavenumbers, labels, label_names, metadata = load_npz_dataset(
    file_path="goldie_graphitic_synthetic_dataset.npz",
    spectra_key="spectra",
    wavenumbers_key="wavenumbers",
    labels_key="y",
    label_names_key="label_names"
)

# Preprocess (if already aligned)
target_grid, processed_spectra = preprocess_aligned_spectra(
    spectra=spectra,
    wavenumbers=wavenumbers,
    baseline_correct=True,
    normalize=True
)

# Split for training
X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
    processed_spectra, labels
)
```

## Functions Added

### `load_npz_dataset()`
Loads data from `.npz` file with flexible key names.

**Parameters:**
- `file_path`: Path to `.npz` file
- `spectra_key`: Key name for spectra (default: `"spectra"`)
- `wavenumbers_key`: Key name for wavenumbers (default: `"wavenumbers"`)
- `labels_key`: Key name for labels (default: `"y"`)
- `label_names_key`: Key name for label names (default: `"label_names"`)

**Returns:**
- `spectra`: Array of shape `(n_samples, n_wavenumbers)`
- `wavenumbers`: Array of shape `(n_wavenumbers,)` or `(n_samples, n_wavenumbers)`
- `labels`: Array of integer labels
- `label_names`: Array of label name strings (or None)
- `metadata`: Dictionary with additional keys (e.g., `id_ig`, `i2d_ig`)

### `convert_npz_to_list_format()`
Converts numpy arrays to list format (for compatibility with existing preprocessing functions).

### `preprocess_aligned_spectra()`
Efficient preprocessing for already-aligned spectra (when all spectra share the same wavenumber grid).

**Use this when:**
- Spectra are already on the same wavenumber grid
- You want faster preprocessing (no alignment step)

**Parameters:**
- `spectra`: Array of shape `(n_samples, n_wavenumbers)`
- `wavenumbers`: Array of shape `(n_wavenumbers,)`
- `align`: Set to `False` if already aligned (default: `False`)
- Other preprocessing options same as `preprocess_dataset()`

## Example: Your Data Format

Based on your data structure:

```python
import numpy as np
from data_ingestion import load_npz_dataset
from preprocessing import preprocess_aligned_spectra

# Load your data
data = np.load("goldie_graphitic_synthetic_dataset.npz")
spectra = data["spectra"]          # X
wavenumbers = data["wavenumbers"]  # x_axis
labels = data["y"]                 # y
label_names = data["label_names"]  # labels
id_ig = data["id_ig"]             # D/G ratios
i2d_ig = data["i2d_ig"]           # 2D/G ratios

# Or use the helper function:
spectra, wavenumbers, labels, label_names, metadata = load_npz_dataset(
    "goldie_graphitic_synthetic_dataset.npz"
)
# metadata will contain id_ig and i2d_ig

# Preprocess
target_grid, processed_spectra = preprocess_aligned_spectra(
    spectra=spectra,
    wavenumbers=wavenumbers,
    baseline_correct=True,
    normalize=True,
    normalize_method='max'
)
```

## Integration with Training

After loading and preprocessing, you can use the data directly with the training pipeline:

```python
from model import create_model
from training import train_model, create_dataloader

# Create model
model = create_model(
    input_length=len(target_grid),
    n_classes=len(np.unique(labels))
)

# Create data loaders
train_loader = create_dataloader(X_train, y_train, batch_size=32)
val_loader = create_dataloader(X_val, y_val, batch_size=32)

# Train (see example_workflow.py for full training code)
```

## Notes

- If your spectra are **already aligned** (same wavenumber grid), use `preprocess_aligned_spectra()` for better performance
- If your spectra have **different wavenumber grids**, use `convert_npz_to_list_format()` + `preprocess_dataset()`
- The `metadata` dictionary contains all additional keys from your `.npz` file (like `id_ig`, `i2d_ig`)
- Label names are automatically converted to strings if stored as object arrays

## Troubleshooting

**Issue**: "Key not found" error
- **Solution**: Check that your `.npz` file has the expected keys, or specify custom key names in `load_npz_dataset()`

**Issue**: Shape mismatch errors
- **Solution**: Ensure `spectra` is 2D `(n_samples, n_wavenumbers)` and `labels` is 1D `(n_samples,)`

**Issue**: Wavenumbers shape issues
- **Solution**: If all spectra share the same grid, ensure `wavenumbers` is 1D. If different, it should be 2D `(n_samples, n_wavenumbers)`


