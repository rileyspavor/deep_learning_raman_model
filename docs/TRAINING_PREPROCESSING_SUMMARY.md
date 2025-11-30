# Training Preprocessing Summary

## Analysis of Actual Training Data

Based on analysis of `training_data/train_val_test.npz`:

### D. Range
✅ **Trimmed to 800-3200 cm⁻¹**
- Target grid: 800.0 - 3200.0 cm⁻¹
- Length: 1500 points
- Step size: ~1.6 cm⁻¹
- **Interpolated exactly to `target_grid.npy`**

### A. Baseline Correction
❌ **NOT APPLIED** (or applied incorrectly)
- Training data contains **negative values** (min: -373.93)
- This suggests baseline correction was either:
  - Not applied during training
  - Applied incorrectly
  - Data was loaded from a source that didn't have baseline correction

**Expected:** Asymmetric Least Squares (ALS) with:
- `lam=1e4` (smoothness parameter)
- `p=0.001` (asymmetry parameter)
- `n_iter=10` (iterations)

### B. Normalization
❌ **NOT APPLIED**
- Max value in training data: **3375.65** (not ~1.0)
- Mean: 155.41 (not ~0.0)
- Std: 274.71 (not ~1.0)

**Expected:** Max normalization (divide by maximum intensity)

### C. Smoothing
❓ **UNCERTAIN** (cannot detect from final data)
- Code suggests `smooth=False` in most scripts
- But `process_multiple_datasets.py` has `smooth=True`
- Need to check which script was actually used

---

## ⚠️ CRITICAL ISSUE DETECTED

**Your inference script applies preprocessing, but your training data was NOT preprocessed!**

This is a **major mismatch** that explains:
- Why model confidence is low (16.9%)
- Why predictions are uncertain
- Why the model can't distinguish classes well

### The Problem:
1. **Training data:** Raw spectra (no baseline correction, no normalization)
2. **Inference:** Preprocessed spectra (baseline correction + normalization)

The model was trained on one format but is receiving a different format!

---

## What Actually Happened During Training?

Based on the data analysis, it appears:
1. ✅ Spectra were aligned to 800-3200 cm⁻¹ grid (1500 points)
2. ❌ Baseline correction was **NOT** applied
3. ❌ Normalization was **NOT** applied
4. ❓ Smoothing is uncertain

This suggests the training data may have been:
- Loaded from a pre-existing `.npz` file that didn't have preprocessing
- Or preprocessing was skipped during training
- Or a different preprocessing pipeline was used

---

## Solution: Match Inference to Training

You have two options:

### Option 1: Retrain with Preprocessing (RECOMMENDED)
Apply the same preprocessing during training that you use in inference:
- Baseline correction: ALS
- Normalization: Max
- Smoothing: None

### Option 2: Remove Preprocessing from Inference
Match inference to what was actually used in training:
- No baseline correction
- No normalization
- Just alignment to target_grid

---

## Code References

Training scripts show these settings were intended:
- `example_workflow.py`: `baseline_correct=True, normalize=True, smooth=False`
- `step_by_step_preprocessing.py`: `baseline_correct=True, normalize=True, smooth=False`
- `process_multiple_datasets.py`: `baseline_correct=True, normalize=True, smooth=True`

But the actual training data doesn't match these settings!



