# Walkthrough - Test-Time Augmentation (TTA) for nnUNet v2 Inference

We have implemented a standalone script [predict_from_raw_data_TTA.py] that allows calculating aleatoric uncertainty (variance and Shannon entropy) and pairwise prediction diversity (DSC) via Test-Time Augmentation (TTA) during inference without modifying any existing nnUNet scripts.

---

## Architecture & Implementation Overview

### 1. Custom Prediction Script

A new script [predict_from_raw_data_TTA.py] has been created under `nnunetv2/inference/`. It defines `nnUNetPredictorTTA`, a subclass of `nnUNetPredictor` that overrides key methods to perform 10-pass Test-Time Augmentation and uncertainty quantification.

### 2. TTA Transforms (using MONAI)

A list of 10 transforms is applied to each input volume:

1. `"original"`: Identity / no transformation.
2. `RandRotated(keys=["img"], prob=1, range_x=5*pi/180, range_y=5*pi/180, range_z=10*pi/180, mode="bilinear", padding_mode="border", lazy=False)`
3. `RandRotated(keys=["img"], prob=1, range_x=3*pi/180, range_y=3*pi/180, range_z=5*pi/180, mode="bilinear", padding_mode="border", lazy=False)`
4. `RandZoomd(keys=["img"], prob=1, min_zoom=0.9, max_zoom=1.1, mode="bilinear", padding_mode="edge", lazy=False)`
5. `RandZoomd(keys=["img"], prob=1, min_zoom=0.95, max_zoom=1.05, mode="bilinear", padding_mode="edge", lazy=False)`
6. `RandFlipd(keys=["img"], prob=1, spatial_axis=2)`
7. `RandGaussianNoised(keys=["img"], prob=1, std=0.01)`
8. `RandGaussianNoised(keys=["img"], prob=1, std=0.03)`
9. `RandAdjustContrastd(keys=["img"], prob=1, gamma=(0.9, 1.1))`
10. `RandAdjustContrastd(keys=["img"], prob=1, gamma=(0.95, 1.05))`

### 3. Forward Pass & Inversion Pipeline

- Input volume is ensured to be a 4D tensor `(C, W, H, D)` compatible with MONAI dictionary transforms.
- Invertible spatial transforms (rotations, zooms, flips) are inverted back to original image coordinates using MONAI's `Invertd` with `nearest_interp=False` to preserve continuous probability values.
- Intensity transforms (noise, contrast adjustments) maintain `aligned_prob = aug_data["pred"]`.

### 4. Ensemble Aggregation & Uncertainty Estimation

Stack all passes into `all_probs` (shape: `(N_TTA, C, W, H, D)`):

- **Mean Probabilities**: $\mu = \frac{1}{N_{TTA}} \sum_{i=1}^{N_{TTA}} p_i$
- **Variance Map**: Average variance across classes: $\text{Var}_{\text{map}} = \frac{1}{C} \sum_{c=1}^C \text{Var}(p_{i, c})$
- **Entropy Map**: Shannon entropy across classes: $H = - \sum_{c=1}^C \mu_c \log(\mu_c + 10^{-8})$
- **Pairwise DSC Diversity**: Computes pairwise Dice Similarity Coefficient between each pair $(i, j)$ of individual TTA predictions (`all_probs[i] > 0.5` vs `all_probs[j] > 0.5`) across foreground classes. Saves `pairwise_dscs`, `mean_dscs_per_class`, and `mean_dsc_overall`.

### 5. Resampling & Output Export

- Uses nnUNet's native resampling pipeline (`export_prediction_from_logits` and `export_uncertainty`) to map predictions and uncertainty volumes back to the original patient space (reversing cropping, transposing, and spacing changes).
- Saves all files matching the target dataset naming convention:
  - Segmentation mask: `{patient_id}_{nodule_id}_{mask_name}.nii.gz`
  - Pairwise DSC metrics: `{patient_id}_{nodule_id}_{mask_name}_tta_dsc.json`
  - Entropy uncertainty: `{patient_id}_{nodule_id}_{mask_name}_uncertainty_entropy.nii.gz`
  - Variance uncertainty: `{patient_id}_{nodule_id}_{mask_name}_uncertainty_variance.nii.gz`

---

## CLI Execution Instructions

To run the analysis:

```bash
python nnunetv2/inference/predict_from_raw_data_TTA.py \
    -i <INPUT_FOLDER> \
    -o <OUTPUT_FOLDER> \
    -d <DATASET_NAME_OR_ID> \
    -c 3d_fullres \
    -f 0 \
    -device cuda
```

Example for Dataset001_LungNodule:

```bash
python nnunetv2/inference/predict_from_raw_data_TTA.py \
    -i /home/kdovrou/PhD/models/nnUnet_raw/Dataset001_LungNodule/imagesTs \
    -o /home/kdovrou/PhD/models/nnUnet_results/pred_nnUnet_test_TTA \
    -d 1 \
    -c 3d_fullres \
    -f 0 \
    -uncertainty_metrics variance entropy
```

### Parameters:

- `-i`: Path to the input test images folder (`imagesTs`).
- `-o`: Output directory for predicted masks and uncertainty maps.
- `-d`: Dataset ID or name.
- `-c`: Configuration name (e.g. `3d_fullres`).
- `-f`: Folds to use for prediction (e.g. `0` or `0 1 2 3 4`).
- `-uncertainty_metrics`: Choose which uncertainty metrics to calculate and save (`variance`, `entropy`, or both). Default: `variance entropy`.
- `--save_probabilities`: Optional flag to export predicted class probabilities.
- `-prob_folder`: Optional folder where probability maps will be saved (used with `--save_probabilities`).
- `--disable_uncertainty`: Optional flag to disable exporting uncertainty maps.
- `--disable_tta`: Optional flag to disable inner sliding-window tile mirroring for faster execution.
- `-npp` / `-nps`: Number of processes for preprocessing / export (set `-npp 0 -nps 0` for non-multiprocessing mode).
