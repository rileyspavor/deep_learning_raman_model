# Next Steps Guide

## Quick Start: Test the System

### Step 1: Run the Example Workflow
Test the complete pipeline with synthetic data:

```bash
python example_workflow.py
```

This will:
- Generate synthetic Raman spectra
- Preprocess the data
- Train a model
- Evaluate performance
- Save the trained model

**Expected output:** Training progress, metrics, and saved model files.

---

## Prepare Your Real Data

### Step 2: Organize Your Data

You have two options for organizing your Raman spectra:

#### Option A: Directory Structure (Recommended)
```
your_data/
├── graphite/
│   ├── spectrum1.txt
│   ├── spectrum2.txt
│   └── ...
├── graphene/
│   ├── spectrum1.txt
│   └── ...
├── GO/
└── rGO/
```

#### Option B: Single Directory with Label Mapping
Create a JSON file mapping file patterns to labels:
```json
{
  "graphite": 0,
  "graphene": 1,
  "GO": 2,
  "rGO": 3,
  "GNP": 4
}
```

### Step 3: Load Your Data

```python
from data_ingestion import load_raman_dataset

# If using directory structure
wavenumbers_list, intensities_list, labels, filenames = load_raman_dataset(
    data_dir='path/to/your_data',
    file_pattern='*.txt'  # or '*.csv'
)

# If using label mapping
from data_ingestion import load_label_mapping
label_mapping = load_label_mapping('label_mapping.json')
wavenumbers_list, intensities_list, labels, filenames = load_raman_dataset(
    data_dir='path/to/your_data',
    label_mapping=label_mapping
)
```

### Step 4: Check Your Data Format

Your spectrum files should have two columns:
- Column 1: Wavenumber (cm⁻¹)
- Column 2: Intensity

If your files use different column names or formats, adjust the `wavenumber_col` and `intensity_col` parameters in `load_raman_spectrum()`.

---

## Customize Preprocessing

### Step 5: Adjust Preprocessing Parameters

Edit the preprocessing configuration based on your data:

```python
from preprocessing import preprocess_dataset

target_grid, processed_spectra = preprocess_dataset(
    wavenumbers_list=wavenumbers_list,
    intensities_list=intensities_list,
    target_grid=None,  # Auto-infer, or specify: np.arange(800, 3201, 1.0)
    
    # Baseline correction
    baseline_correct=True,  # Set False if your data is already corrected
    baseline_method='als',  # Options: 'als', 'polynomial', 'rolling_ball', 'none'
    
    # Normalization
    normalize=True,
    normalize_method='max',  # Options: 'max', 'area', 'g_peak', 'minmax', 'zscore'
    
    # Smoothing (optional)
    smooth=False,  # Set True if data is noisy
    smooth_method='savgol',  # Options: 'savgol', 'gaussian', 'moving_average'
    
    # Additional parameters
    baseline_lam=1e4,  # For ALS method
    smooth_window_length=5  # For smoothing
)
```

**Tip:** Start with default settings, then adjust based on your data quality.

---

## Train Your Model

### Step 6: Create a Training Script

Create a custom training script (or modify `example_workflow.py`):

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import create_model
from training import train_model, create_dataloader, EarlyStopping
from utils import stratified_split

# Load and preprocess your data (Steps 3-5)
# ... your data loading code ...

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
    processed_spectra, labels,
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
)

# Create data loaders
train_loader = create_dataloader(X_train, y_train, batch_size=32, shuffle=True)
val_loader = create_dataloader(X_val, y_val, batch_size=32, shuffle=False)

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model(
    input_length=len(target_grid),
    n_classes=len(np.unique(labels)),
    config={
        'n_channels': [32, 64, 128, 256],  # Adjust based on data size
        'dropout': 0.3,
        'use_ordinal_head': False
    }
)
model.to(device)

# Train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    n_epochs=100,
    optimizer=optimizer,
    criterion=criterion,
    device=device,
    scheduler=scheduler,
    early_stopping=early_stopping
)
```

### Step 7: Evaluate Performance

```python
from utils import compute_metrics, plot_confusion_matrix, print_classification_report

# Make predictions
model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test).unsqueeze(1).to(device)
    logits, _ = model(X_test_tensor)
    y_pred = torch.argmax(logits, dim=1).cpu().numpy()

# Compute metrics
class_names = ['graphite', 'graphene', 'GO', 'rGO', 'GNP']  # Your class names
metrics = compute_metrics(y_test, y_pred, class_names=class_names)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Macro F1: {metrics['macro_f1']:.3f}")

# Visualize
plot_confusion_matrix(y_test, y_pred, class_names=class_names)
print_classification_report(y_test, y_pred, class_names=class_names)
```

---

## Use the Model for Inference

### Step 8: Save Your Trained Model

```python
from inference import RamanSpectrumClassifier, save_classifier

# Create classifier wrapper
classifier = RamanSpectrumClassifier(
    model=model,
    target_grid=target_grid,
    class_names={0: 'graphite', 1: 'graphene', 2: 'GO', 3: 'rGO', 4: 'GNP'},
    device=device,
    preprocessing_config={
        'baseline_correct': True,
        'baseline_method': 'als',
        'normalize': True,
        'normalize_method': 'max'
    }
)

# Save for later use
save_classifier(
    classifier,
    model_path='my_trained_model.pth',
    target_grid_path='target_grid.npy',
    class_names_path='class_names.json'
)
```

### Step 9: Use for New Predictions

```python
from inference import load_classifier, load_raman_spectrum

# Load saved model
classifier = load_classifier(
    model_path='my_trained_model.pth',
    target_grid_path='target_grid.npy',
    class_names_path='class_names.json'
)

# Load new spectrum
wavenumbers, intensities = load_raman_spectrum('new_spectrum.txt')

# Predict
result = classifier.predict(wavenumbers, intensities)
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.3f}")

# Quality check
qc_result = classifier.quality_check(
    wavenumbers, intensities,
    confidence_threshold=0.7
)
print(f"QC Passed: {qc_result['qc_passed']}")
```

---

## Advanced: Data Augmentation

### Step 10: Combine Real and Synthetic Data

If you have limited real data, augment with synthetic:

```python
from data_ingestion import generate_synthetic_dataset

# Generate synthetic data
synthetic_spectra, synthetic_labels = generate_synthetic_dataset(
    wavenumber_grid=target_grid,
    n_samples_per_class=100,  # Adjust based on your needs
    class_labels=class_labels
)

# Combine with real data
all_spectra = np.vstack([real_spectra, synthetic_spectra])
all_labels = np.concatenate([real_labels, synthetic_labels])
```

---

## Troubleshooting

### Common Issues:

1. **Import errors**: Make sure you've installed requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. **Out of memory**: Reduce batch size or model size:
   ```python
   batch_size=16  # Instead of 32
   n_channels=[16, 32, 64, 128]  # Smaller model
   ```

3. **Poor performance**: 
   - Try different preprocessing methods
   - Increase training data
   - Adjust model architecture
   - Check data quality

4. **Wavenumber grid mismatch**: Ensure all spectra cover similar ranges or adjust `target_grid`

---

## Recommended Workflow Order

1. ✅ **Test with synthetic data** (`example_workflow.py`)
2. ✅ **Prepare your real data** (organize files, check formats)
3. ✅ **Load and preprocess** your data
4. ✅ **Train model** with your data
5. ✅ **Evaluate** on test set
6. ✅ **Save model** for production use
7. ✅ **Deploy** for inference on new spectra

---

## Next: Customize for Your Use Case

- Adjust class labels to match your materials
- Modify preprocessing based on your instrument/measurement conditions
- Tune model architecture for your data size
- Add custom quality metrics
- Integrate with your lab workflow

Good luck! 🚀


