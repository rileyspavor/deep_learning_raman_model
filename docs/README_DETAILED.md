# Deep Learning Raman Spectroscopy Classification Model (V3) - Detailed Documentation

A comprehensive deep learning framework for classifying graphene materials from Raman spectroscopy data using 1D Convolutional Neural Networks. This document provides in-depth information about the project structure, usage, and implementation details.

##README.md Is meant for TA's to reproduce results


## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Installation](#installation)
4. [Data](#data)
5. [Model Architecture](#model-architecture)
6. [Training](#training)
7. [Testing and Inference](#testing-and-inference)
8. [Visualization Tools](#visualization-tools)
9. [Advanced Usage](#advanced-usage)
10. [Troubleshooting](#troubleshooting)

## Overview

This project implements a 1D Convolutional Neural Network (CNN) for classifying different types of graphene materials based on their Raman spectroscopy signatures. The model can distinguish between 9 different classes of graphene-related materials:

- Graphite
- Graphene Oxide (GO)
- Reduced Graphene Oxide (rGO)
- Defective Graphene
- Multilayer Graphene
- Exfoliated Graphene
- Graphitized Carbon
- GNP (Graphene Nanoplatelets) - High Quality
- GNP (Graphene Nanoplatelets) - Medium Quality

### Key Features

- **Automated Preprocessing**: Handles real-world Raman spectra with varying wavenumber grids
- **Model Versioning**: Automatic versioning system prevents overwriting trained models
- **Comprehensive Visualization**: Multiple tools for visualizing training data, test data, and model predictions
- **Flexible Inference**: Test on single files or entire directories
- **Resume Training**: Continue training from checkpoints with full history preservation

## Project Structure

```
project_root/
├── data/                        # All data folders
│   ├── raw/                     # Original, immutable data
│   │   ├── data/                 # Raw spectrum files (.txt)
│   │   ├── raw_synthetic_data/  # Raw synthetic data files
│   │   └── synthetic_large_npv_data/  # Large synthetic datasets (.npz)
│   ├── processed/               # Data after preprocessing
│   │   ├── v3_data/              # V3 training datasets (.npz)
│   │   │   └── synthetic_graphene_parametric_9class_v2.npz  # FINALIZED dataset
│   │   ├── full_training_raman_labels/  # Full training datasets
│   │   └── training_data/       # Training data with metadata
│   ├── test/                    # Test data
│   │   └── testing_real_data/   # Real Raman spectra for testing (.txt)
│   └── class_labels.json        # Mappings and metadata
├── docs/                        # Documentation (MD and TXT files)
│   ├── MODEL_ARCHITECTURE.txt   # Detailed model architecture explanation
│   ├── GRADIO_DEMO_README.md    # Gradio demo documentation
│   ├── VISUALIZATION_README.md  # Visualization guide
│   ├── V3_TRAINING_README.md    # Training process documentation
│   ├── TRAINING_PREPROCESSING_SUMMARY.md  # Preprocessing details
│   ├── NPZ_FORMAT_GUIDE.md      # Data format specifications
│   └── Synthetic Data Generation Methodology & Justification.txt
├── models/                      # Saved model weights and artifacts
│   └── saved_models_v3/         # Versioned model directories
│       ├── model_v1/            # Model version 1
│       ├── model_v2/            # Model version 2
│       └── model_v3/            # Model version 3 (and newer) #FINAL MODEL
│           ├── model_state_v*.pth      # Model weights (for inference)
│           ├── model_checkpoint_v*.pth # Training checkpoints (weights + optimizer state)
│           ├── class_names_v*.json    # Class name mappings
│           ├── target_grid_v*.npy      # Wavenumber grid used for alignment
│           ├── training_history_v*.json # Training metrics history
│           ├── training_history_v*.png # Training plots (loss, accuracy)
│           └── confusion_matrix_v*.png # Confusion matrix visualization
├── results/                     # Output files and visualizations
│   ├── visualizations/          # CNN visualization outputs
│   │   ├── *_visualization.png  # Full visualization with feature maps
│   │   └── *_activations.png    # Layer activation visualizations
│   ├── training_data_visual_*.png  # Training data visualizations
│   ├── test_data.png            # All test spectra on one plot
│   ├── test_vs_training_*.png   # Test spectrum vs training comparison
│   └── prediction_results_*.txt    # Prediction results on real_test_data
├── src/                         # Core source code
│   ├── model.py                 # CNN model definition (Raman1DCNN, Conv1DBlock)
│   ├── training.py              # Training functions (train_model, validation)
│   ├── data_ingestion.py        # Data loading utilities (load_npz_dataset, load_raman_spectrum)
│   ├── preprocessing.py         # Data preprocessing (normalization, alignment)
│   ├── inference.py             # Inference utilities
│   └── utils.py                 # Helper functions
├── scripts/                     # Standalone scripts
│   ├── train_v3.py              # Main training script
│   ├── test_real_data_v3.py     # Test script for real data
│   ├── plot_training_spectra.py # Visualize training data
│   ├── plot_test_data.py        # Visualize test real data
│   ├── compare_test_to_training.py  # Compare test spectrum to training data
│   ├── visualize_cnn_classification.py  # CNN visualization
│   ├── gradio_demo.py           # Interactive web demo
│   └── ...                      # Other utility scripts
├── requirements.txt             # Python dependencies
└── README.md                    # Simple quick-start guide
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step-by-Step Installation

1. **Clone or download the repository**

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages

The `requirements.txt` includes:
- `torch` - PyTorch for deep learning
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `scipy` - Scientific computing (interpolation, signal processing)
- `pandas` - Data manipulation
- `scikit-learn` - Machine learning utilities
- Additional dependencies for data handling and visualization

## Data

### Data Sources

The project uses synthetic training data generated using parametric models based on known Raman spectroscopy characteristics of graphene materials. Real test data consists of actual Raman spectra collected from various graphene samples.

### Data Format

#### Training Data (.npz format)

The training data is stored in NumPy compressed format (.npz) with the following keys:
- `spectra`: Array of shape (n_samples, n_wavenumbers) - Intensity values
- `wavenumbers`: Array of shape (n_wavenumbers,) - Wavenumber grid in cm⁻¹
- `y`: Array of shape (n_samples,) - Integer class labels
- `label_names`: Array of class name strings

#### Test Data (.txt format)

Real test spectra are stored as tab-separated or space-separated text files with two columns:
- Column 1: Wavenumber (cm⁻¹)
- Column 2: Intensity

Example:
```
1000.0    0.05
1001.0    0.06
1002.0    0.07
...
```

### Data Preprocessing

The model automatically handles:
1. **Wavenumber Alignment**: Interpolates test spectra to match the model's target wavenumber grid
2. **Normalization**: Scales intensities appropriately
3. **Range Matching**: Handles spectra with different wavenumber ranges

## Model Architecture

The model uses a 1D CNN architecture specifically designed for Raman spectroscopy data:

### Architecture Overview

1. **Input Layer**: Accepts 1D spectra (length = number of wavenumber points)
2. **Convolutional Blocks**: Multiple 1D conv layers with batch normalization and ReLU activation
3. **Pooling Layers**: Max pooling for dimensionality reduction
4. **Fully Connected Layers**: Classification head with dropout for regularization
5. **Output Layer**: Softmax activation for class probabilities

### Key Design Decisions

- **1D Convolutions**: Appropriate for sequential spectral data
- **Batch Normalization**: Stabilizes training and improves convergence
- **Dropout**: Prevents overfitting
- **Multiple Convolutional Layers**: Captures features at different scales
- **Global Pooling**: Reduces spatial dimensions before classification

For detailed architecture information, see `docs/MODEL_ARCHITECTURE.txt`.

## Training

### Basic Training

Train a new model from scratch:

```bash
python scripts/train_v3.py --data "data/processed/v3_data/synthetic_graphene_parametric_9class_v2.npz"
```

This will:
- Create a new versioned model directory (e.g., `model_v4`)
- Train for 100 epochs (default)
- Save checkpoints periodically
- Generate training history plots
- Save the final model state

### Training Parameters

- **Epochs**: Number of training iterations (default: 100)
- **Batch Size**: Number of samples per batch (default: 32)
- **Learning Rate**: Initial learning rate (default: 0.001)
- **Early Stopping**: Automatically stops if validation loss doesn't improve

### Resuming Training

Continue training from a previous checkpoint:

```bash
# Continue from a specific version
python scripts/train_v3.py --data "data/processed/v3_data/synthetic_graphene_parametric_9class_v2.npz" --pretrained-version 4

# Shortcut syntax
python scripts/train_v3.py --data "data/processed/v3_data/synthetic_graphene_parametric_9class_v2.npz" --from-checkpoint 3
```

When resuming:
- Model weights are loaded from checkpoint
- Optimizer state is restored
- Training history is continued
- Epoch numbering is preserved

### Model Versioning

Each training run creates a new versioned directory:
- `model_v1/`, `model_v2/`, `model_v3/`, etc.
- Prevents accidental overwriting
- Easy to compare different training runs
- Latest version is automatically detected

### Training Outputs

Each model version directory contains:
- `model_state_v*.pth`: Model weights for inference
- `model_checkpoint_v*.pth`: Full checkpoint (weights + optimizer state)
- `class_names_v*.json`: Class name mappings
- `target_grid_v*.npy`: Wavenumber grid used
- `training_history_v*.json`: Training metrics (loss, accuracy per epoch)
- `training_history_v*.png`: Visualization of training curves
- `confusion_matrix_v*.png`: Classification performance matrix

## Testing and Inference

### Testing on Real Data

Test the model on real Raman spectra:

```bash
# Test all files in default directory
python scripts/test_real_data_v3.py

# Test a specific file
python scripts/test_real_data_v3.py --file "G-1 With Peaks.txt"

# Test with specific model version
python scripts/test_real_data_v3.py --model-dir models/saved_models_v3/model_v3
```

### Output Formats

#### Text Output (default)
- Detailed prediction for each file
- All class probabilities
- Summary section with model information
- Saved to `results/prediction_results_N.txt` (auto-incremented)

#### CSV Output
```bash
python scripts/test_real_data_v3.py --csv
```
- Machine-readable format
- Includes file path, predicted class, confidence, all probabilities
- Saved to `results/prediction_results_N.csv`

### Interactive Model Selection

If no model is specified, the script will:
1. List all available model versions
2. Prompt you to select one
3. Load the selected model

### Prediction Details

For each test file, the model provides:
- **Predicted Class**: The most likely class
- **Confidence**: Probability of the predicted class
- **All Probabilities**: Full probability distribution across all classes
- **Top Alternatives**: Other likely classes (if confidence is low)

## Visualization Tools

### 1. Visualize Training Data

Plot random spectra from each class in the training dataset:

```bash
# Interactive mode - select dataset and number of samples
python scripts/plot_training_spectra.py

# Use specific dataset
python scripts/plot_training_spectra.py --data-file "data/processed/v3_data/synthetic_graphene_parametric_9class_v2.npz" --n-samples 5
```

**Features:**
- Select from available datasets interactively
- Choose number of random spectra per class
- Overlays multiple spectra per class
- Auto-generates filename based on dataset name
- Saves to `results/training_data_visual_{dataset_name}.png`

### 2. Visualize Test Data

Plot all test spectra on a single plot:

```bash
python scripts/plot_test_data.py
```

**Features:**
- Shows all test files simultaneously
- Different colors for each spectrum
- Legend or count indicator
- Saves to `results/test_data.png`

### 3. Compare Test to Training

Visualize how a test spectrum compares to training data:

```bash
# Interactive mode - select test file
python scripts/compare_test_to_training.py

# Use specific file
python scripts/compare_test_to_training.py --file "G-1 With Peaks.txt"
```

**Features:**
- Runs test spectrum through model
- Shows prediction and confidence
- Loads training spectra from predicted class
- Overlays test spectrum (red) on training spectra (blue)
- Visual comparison of similarity
- Saves to `results/test_vs_training_{filename}.png`

### 4. CNN Visualization

Visualize how the CNN processes a spectrum:

```bash
python scripts/visualize_cnn_classification.py --file "G-1 With Peaks.txt"
```

**Features:**
- Shows input spectrum
- Displays feature maps from each layer
- Shows layer activations
- Visualizes the classification process
- Saves detailed visualizations to `results/visualizations/`

## Advanced Usage

### Custom Model Configuration

Modify model architecture in `src/model.py`:
- Number of convolutional layers
- Filter sizes and counts
- Pooling strategies
- Dropout rates
- Fully connected layer sizes

### Custom Data Loading

Extend `src/data_ingestion.py` to:
- Support additional file formats
- Implement custom preprocessing
- Add data augmentation
- Handle different data structures

### Batch Processing

Process multiple files programmatically:

```python
from scripts.test_real_data_v3 import test_real_data, load_v3_model

# Load model once
model, target_grid, class_names, _, _ = load_v3_model("models/saved_models_v3")

# Process multiple files
files = ["file1.txt", "file2.txt", "file3.txt"]
for file in files:
    result = test_real_data(file, model=model, target_grid=target_grid, class_names=class_names)
    print(f"{file}: {result['predicted_class']} ({result['confidence']:.1%})")
```

### Integration with Other Tools

The model can be integrated into:
- Jupyter notebooks for interactive analysis
- Web applications (see `scripts/gradio_demo.py`)
- Automated pipelines
- Custom workflows

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**: Ensure you're running scripts from the project root directory, or the scripts should automatically add the project root to `sys.path`.

#### 2. Model File Not Found

**Problem**: `FileNotFoundError: Model file not found`

**Solution**: 
- Check that model directory exists: `models/saved_models_v3/model_v3/`
- Verify model files are present
- Use `--model-dir` to specify correct path
- List available models interactively

#### 3. Data Loading Errors

**Problem**: Errors loading .txt spectrum files

**Solution**:
- Check file format (should be two columns: wavenumber, intensity)
- Verify file encoding (UTF-8 recommended)
- Ensure no missing values or corrupted data
- Check file path is correct

#### 4. Wavenumber Range Mismatch

**Problem**: Test spectrum wavenumber range doesn't match training data

**Solution**: The model automatically handles this through interpolation. If issues persist:
- Check that test spectrum has sufficient overlap with training range
- Verify wavenumber units (should be cm⁻¹)
- Ensure data is not corrupted

#### 5. Low Prediction Confidence

**Problem**: Model predictions have low confidence

**Possible Causes**:
- Test spectrum is significantly different from training data
- Spectrum quality is poor (high noise, artifacts)
- Wavenumber range mismatch
- Class not well-represented in training data

**Solutions**:
- Check spectrum quality
- Compare to training data using visualization tools
- Consider retraining with more diverse data
- Verify preprocessing is correct

### Performance Tips

1. **GPU Acceleration**: Training is faster on GPU. Ensure CUDA is available:
   ```python
   import torch
   print(torch.cuda.is_available())
   ```

2. **Batch Size**: Increase batch size for faster training (if memory allows)

3. **Data Loading**: Use multiple workers for data loading during training

4. **Model Caching**: Load model once and reuse for multiple predictions

### Getting Help

- Check `docs/` directory for detailed documentation
- Review error messages carefully - they often indicate the issue
- Use visualization tools to inspect data quality
- Verify data formats match expected structure

## Additional Resources

- **Model Architecture**: `docs/MODEL_ARCHITECTURE.txt`
- **Training Details**: `docs/V3_TRAINING_README.md`
- **Data Format**: `docs/NPZ_FORMAT_GUIDE.md`
- **Visualization Guide**: `docs/VISUALIZATION_README.md`
- **Gradio Demo**: `docs/GRADIO_DEMO_README.md`

## License and Citation

[Add your license and citation information here]

## Contact

Riley Spavor

riley.spavor@gmail.com

