# Walkthrough - Monte Carlo Dropout for nnUNet v2 Inference

We have implemented a standalone script [predict_from_raw_data_mc.py](file:///c:/Users/aikat/Documents/PhD/models/nnUNet/nnunetv2/inference/predict_from_raw_data_mc.py) that allows calculating epistemic uncertainty (variance and entropy) during inference without modifying any existing nnUNet scripts.

## Changes Made

### 1. Created Custom Prediction Script
A new script [predict_from_raw_data_mc.py](file:///c:/Users/aikat/Documents/PhD/models/nnUNet/nnunetv2/inference/predict_from_raw_data_mc.py) has been created under `nnunetv2/inference/`. It defines `nnUNetPredictorMC`, a subclass of `nnUNetPredictor` that overrides key methods to perform Monte Carlo dropout sampling and uncertainty quantification.

### 2. Implemented MC Dropout Logic
- **Dynamic Injection**: We implemented `inject_mc_dropout()` which finds the `ConvDropoutNormReLU` building blocks used inside standard nnUNet UNet architectures (both plain and residual) and sets their `dropout` modules dynamically. It also provides a recursive fallback that injects dropout after activation functions (`LeakyReLU`, `ReLU`, `ELU`) if standard blocks are not found.
- **Interception of `.eval()`**: We patched the network's `eval()` method so that calling it puts the overall model in evaluation mode (important for batch normalization / instance normalization behaviors), but immediately sets all dropout modules to `train(True)` mode, maintaining active dropout during sliding window inference.
- **Averaging and Uncertainty Quantification**:
  - Probability predictions are calculated for $N$ Monte Carlo passes (softmax/sigmoid outputs).
  - The mean probability map is converted back to pseudo-logits (`log(mean_probs + 1e-8)`) to preserve compatibility with the rest of nnUNet's label mapping and resampling pipeline.
  - Epistemic uncertainty maps are computed as **variance** (average variance across classes) and **entropy** (predictive entropy of the mean probability).
- **Float32 Resampling & Saving**: We implemented a robust NIfTI writer (`write_float_image`) and resampling logic (`export_uncertainty`) to save the variance and entropy maps as high-precision float32 NIfTI files (`.nii.gz`) matching the original spacing, affine, and origin metadata.

## CLI Execution Instructions

Since you are editing files locally and running them on your remote server, sync the changes to your remote server (`Thorax`), and then run the inference command.

The script has the exact same command-line interface as the standard `nnUNetv2_predict` tool, with additional arguments for Monte Carlo dropout:

```bash
python nnunetv2/inference/predict_from_raw_data_mc.py \
    -i <INPUT_FOLDER> \
    -o <OUTPUT_FOLDER> \
    -d <DATASET_NAME_OR_ID> \
    -c 3d_fullres \
    -f <FOLDS> \
    -mc_samples 10 \
    -dropout_prob 0.3 \
    -uncertainty_metrics variance entropy
```

### Additional Parameters Added:
- `-mc_samples`: The number of Monte Carlo runs to execute (default: `10`).
- `-dropout_prob`: The dropout probability (default: `0.3`).
- `-uncertainty_metrics`: Choose which uncertainty metrics to save (`variance`, `entropy`, or both. Default: `variance entropy`).
- `--disable_uncertainty`: Optional flag to disable exporting the uncertainty maps.

### Outputs Created:
1. **Segmentation Output** (e.g. `subject_001.nii.gz`): The class labels predicted from the mean probability map across the Monte Carlo passes.
2. **Uncertainty Variance Map** (e.g. `subject_001_uncertainty_variance.nii.gz`): A float32 NIfTI volume representing the voxel-wise average variance.
3. **Uncertainty Entropy Map** (e.g. `subject_001_uncertainty_entropy.nii.gz`): A float32 NIfTI volume representing the voxel-wise predictive entropy.
