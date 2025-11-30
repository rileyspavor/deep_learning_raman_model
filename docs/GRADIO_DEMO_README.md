# Interactive Web Demo - Gradio Interface

This guide explains how to run the interactive web demo for the Raman spectroscopy classifier.

## Overview

The Gradio demo provides a browser-based interface where you can:
- Upload Raman spectrum files (.npy, .txt, .csv)
- Paste spectrum arrays directly
- View predictions with confidence scores
- See class probabilities for all 9 classes
- Visualize the spectrum with prediction overlay

## Prerequisites

1. **Trained Model**: You need to train the v3 model first:
   ```bash
   python train_v3.py --data "v3 data/synthetic_graphene_parametric_9class_v2.npz"
   ```
   This will create a `saved_models_v3/` directory with the trained model.

2. **Install Gradio**: 
   ```bash
   pip install gradio>=4.0.0
   ```
   Or install all requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Demo

### Basic Usage

```bash
python gradio_demo.py
```

This will:
- Load the model from `saved_models_v3/`
- Start a local web server
- Open your browser to `http://127.0.0.1:7860`

### Custom Options

```bash
python gradio_demo.py \
    --model-dir "saved_models_v3" \
    --server-name "127.0.0.1" \
    --server-port 7860 \
    --share
```

**Options:**
- `--model-dir`: Directory containing saved model files (default: `saved_models_v3`)
- `--server-name`: Server address (default: `127.0.0.1` for local)
- `--server-port`: Port number (default: `7860`)
- `--share`: Create a public Gradio link (great for sharing!)

## Input Formats

### File Upload
The demo accepts three file formats:

1. **NumPy (.npy)**: Direct numpy array file
   ```python
   np.save("spectrum.npy", your_spectrum_array)
   ```

2. **Text (.txt)**: Space or tab-separated values
   - Single column: intensities only
   - Two columns: wavenumber, intensity (intensities will be extracted)

3. **CSV (.csv)**: Comma-separated values
   - Single column: intensities only
   - Two columns: wavenumber, intensity (intensities will be extracted)

### Paste Array
You can also paste spectrum values directly:
- Comma-separated: `100.5, 102.3, 98.7, ...`
- Space-separated: `100.5 102.3 98.7 ...`

## Expected Format

Your input spectrum should:
- **Length**: 1500 points
- **Wavenumber range**: 800 - 3200 cm⁻¹
- **Format**: 1D array of intensities
- **Data type**: Numeric values (float or int)

## Supported Classes

The model can classify 9 graphene-related materials:

1. **Graphite**
2. **Exfoliated Graphene**
3. **GNP (High Quality)** - Graphene Nanoplatelets
4. **GNP (Medium Quality)** - Graphene Nanoplatelets
5. **Multilayer Graphene**
6. **Graphene Oxide (GO)**
7. **Reduced Graphene Oxide (rGO)**
8. **Defective Graphene**
9. **Graphitized Carbon**

## Output

For each prediction, you'll see:

1. **Predicted Class**: The most likely class name
2. **Confidence Score**: Probability of the predicted class (0-100%)
3. **Class Probabilities**: Probability distribution across all 9 classes
4. **Visualization**: Plot of the input spectrum with prediction in the title

## Example Usage

### Upload a File
1. Click "Upload File" tab
2. Click "Upload Raman Spectrum" button
3. Select your `.npy`, `.txt`, or `.csv` file
4. View results instantly!

### Paste Array
1. Click "Paste Array" tab
2. Paste your spectrum values (comma or space-separated)
3. Results update automatically

## Deploying Online

### Option 1: Gradio Share (Temporary)
Add `--share` flag to get a public URL:
```bash
python gradio_demo.py --share
```
The URL expires after 72 hours of inactivity.

### Option 2: Hugging Face Spaces
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Upload your code and model files
3. Create a `requirements.txt` with dependencies
4. Your demo will be live at `https://huggingface.co/spaces/your-username/your-space-name`

### Option 3: Self-Hosting
- Deploy on your own server
- Use the `--server-name` and `--server-port` options
- Consider using a reverse proxy (nginx) for production

## Troubleshooting

### "Model file not found"
- Make sure you've trained the model first: `python train_v3.py ...`
- Check that `saved_models_v3/` directory exists
- Verify model files are present:
  - `model_state_v3.pth` or `model_checkpoint_v3.pth`
  - `target_grid_v3.npy`
  - `class_names_v3.json`

### "Spectrum length doesn't match"
- Your spectrum must have exactly 1500 points
- Check your input file format
- The model expects wavenumber range 800-3200 cm⁻¹

### Port Already in Use
- Change the port: `--server-port 7861`
- Or stop the existing process using port 7860

## Tips

1. **File Size**: Large files may take longer to upload
2. **Format**: Prefer `.npy` format for fastest loading
3. **Quality**: Higher confidence scores (>80%) indicate more reliable predictions
4. **Visualization**: Check the plot to verify your spectrum looks correct

## Demo Screenshot Features

The demo provides a clean, professional interface similar to:
- Interactive file upload
- Real-time predictions
- Probability bars for all classes
- Spectrum visualization
- Mobile-responsive design

Perfect for presentations, demos, or sharing with collaborators!



