### Step 0

Installation and setup

```bash
export nnUNet_raw="your_path/nnUnet_raw"
export nnUNet_preprocessed="your_path/nnUNet_preprocessed"
export nnUNet_results="your_path/nnUnet_results"
```

### Step 1

Prepare dataset

```
nnUnet_raw/
└── DatasetXXX_Name
  └── imagesTr
    ├── case_1_0000.nii.gz
    ├── case_2_0000.nii.gz
  └── imagesTs
  └── labelsTr
    ├── case_1.nii.gz
    ├── case_2.nii.gz
nnUnet_preprocessed/
nnUnet_results/
```

- Create dataset.json \
  Example:

  ```
  {
    "channel_names": {
      "0": "noNorm" # for already preprocessed data
     },
    "labels": {
      "background": 0,
      "lung_nodule": 1
     },
    "numTraining": number_of_train_val_samples,
    "file_ending": ".nii.gz"
  }
  ```

- Create splits_final.json

```
  [
    {
    "train": [
      case_1,
      case_2
    ],
    "val": [
      case_3,
      case_4
    ]
    }
  ]
```

### Step 2

Plan and preprocess

```
nnUNetv2_plan_and_preprocess -d DATASET_ID -c 3d_fullres --verify_dataset_integrity

nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres --verify_dataset_integrity
```

### Step 3

Train model

```
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres 0

CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 1 3d_fullres 0
```

### Step 4

Run inference

```
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION

nnUNetv2_predict -i imagesTs -o pred_nnUnet -d 1 -c 3d_fullres -f 0
```