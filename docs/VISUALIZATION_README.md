# CNN Classification Visualization Component

This component provides comprehensive visualization tools to understand how the CNN processes and classifies Raman spectroscopy spectra.

## Features

The visualization component shows:

1. **Input Spectrum**: The original spectrum aligned to the target wavenumber grid
2. **Classification Results**: Bar chart showing probabilities for all classes
3. **Feature Maps**: Visualizations of feature maps from each convolutional block
4. **Layer Activations**: Detailed view of how activations flow through the network

## Usage

### Command Line

```bash
# Visualize a single spectrum
python visualize_cnn_classification.py \
    --spectrum-file "data/test/testing_real_data/G-1 With Peaks.txt" \
    --model-dir "saved_models_v3" \
    --output-dir "visualizations"

# Visualize all spectra in a directory (batch mode)
python visualize_cnn_classification.py \
    --spectrum-file "data/test/testing_real_data" \
    --model-dir "saved_models_v3" \
    --output-dir "visualizations" \
    --batch \
    --no-show
```

### Python API

```python
from visualize_cnn_classification import visualize_classification

# Visualize a single spectrum
visualize_classification(
        spectrum_file="data/test/testing_real_data/G-1 With Peaks.txt",
    model_dir="saved_models_v3",
    output_dir="visualizations",
    show_plot=True,
    max_feature_maps=8
)

# Batch mode - process all files in directory
visualize_classification(
        spectrum_file="data/test/testing_real_data",
    model_dir="saved_models_v3",
    output_dir="visualizations",
    show_plot=False,
    batch_mode=True
)
```

### Advanced Usage

```python
from visualize_cnn_classification import CNNVisualizer
from test_real_data_v3 import load_v3_model
import torch

# Load model
model, target_grid, class_names = load_v3_model("saved_models_v3")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create visualizer
visualizer = CNNVisualizer(model, target_grid, class_names, device)

# Visualize classification
results = visualizer.visualize_classification(
        spectrum_file="data/test/testing_real_data/G-1 With Peaks.txt",
    output_path="output.png",
    show_plot=True
)

# Visualize layer activations
results = visualizer.visualize_layer_activations(
        spectrum_file="data/test/testing_real_data/G-1 With Peaks.txt",
    output_path="activations.png",
    show_plot=True
)
```

## Output Files

The visualization component generates two types of output:

1. **Comprehensive Visualization** (`*_visualization.png`):
   - Input spectrum
   - Classification probabilities
   - Feature maps from each convolutional block

2. **Layer Activations** (`*_activations.png`):
   - Input spectrum
   - Averaged activations from each convolutional block
   - Shows how features are extracted at different levels

## Parameters

- `spectrum_file`: Path to spectrum file or directory
- `model_dir`: Directory containing saved model (default: "saved_models_v3")
- `output_dir`: Directory to save visualizations (optional)
- `show_plot`: Whether to display plots (default: True)
- `batch_mode`: Process all files in directory (default: False)
- `max_feature_maps`: Maximum feature maps to show per layer (default: 8)

## Example Output

The visualization shows:

1. **Top Panel**: Input spectrum with wavenumber axis
2. **Second Panel**: Classification results with probabilities for all 9 classes
3. **Bottom Panels**: Feature maps from each of the 4 convolutional blocks

Each feature map panel shows:
- Multiple feature maps (up to `max_feature_maps`)
- Wavenumber axis aligned to the layer's resolution
- Different colors for different feature maps

## Requirements

- torch >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- scipy >= 1.10.0
- pandas >= 2.0.0 (for loading spectrum files)

## Notes

- The visualization automatically handles different spectrum file formats
- Feature maps are shown at their native resolution (after pooling)
- The component uses forward hooks to capture intermediate activations
- GPU acceleration is automatically used if available

