### Step 1

Prepare dataset

nnUnet_raw \
&emsp;    DatasetXXX_Name \
&emsp;&emsp;&emsp;   imagesTr \
&emsp;&emsp;&emsp;&emsp;  case_1_0000.nii.gz \
&emsp;&emsp;&emsp;&emsp;  case_2_0000.nii.gz \
&emsp;&emsp;&emsp;   imagesTs \
&emsp;&emsp;&emsp;   labelsTr \
&emsp;&emsp;&emsp;&emsp;  case_1.nii.gz \
&emsp;&emsp;&emsp;&emsp;  case_2.nii.gz \
nnUnet_preprocessed \
nnUnet_results\ 

* Create dataset.json
Example: \
{ \
&emsp;    "channel_names": { \
&emsp;&emsp;        "0": "noNorm"     # for already preprocessed data \
&emsp;    }, \
&emsp;    "labels": { \
&emsp;&emsp;  "background": 0, \
&emsp;&emsp;  "lung_nodule": 1 \
&emsp;    }, \
&emsp;    "numTraining": number_of_train_val_samples, \
&emsp;    "file_ending": ".nii.gz" \
}

* Create splits_final.json \
[ \
&emsp; { \
&emsp;&emsp;  "train": [ \
&emsp;&emsp;&emsp;  case_1, \
&emsp;&emsp;&emsp;  case_2 \
&emsp;&emsp;  ], \
&emsp;&emsp;   "val": [ \
&emsp;&emsp;&emsp;  case_3, \
&emsp;&emsp;&emsp;  case_4 \
&emsp;&emsp;  ] \
&emsp;    } \
] 


### Step 2 

Plan and preprocess
> nnUNetv2_plan_and_preprocess -d DATASET_ID -c 3d_fullres --verify_dataset_integrity

> nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres --verify_dataset_integrity

### Step 3

Train model

> CUDA_VISIBLE_DEVICES=1 nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres 0

>  CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 1 3d_fullres 0 