import os
from pathlib import Path
from monai.transforms import ScaleIntensityRanged
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
import pandas as pd
import SimpleITK as sitk
import shutil
import json

def convert_data(data: pd.DataFrame, count:int, path_images: Path | str, path_labels: Path | str=None):
    mapping = {
        'dataset': [],
        'nodule': [],
        'mask': [],
        'image_new_name': [],
        'mask_new_name': []
    }

    # read files
    for idx, row in data.iterrows():
        image = sitk.ReadImage(path_base.joinpath(f"Data/{row['image_path']}"))
        image_array = sitk.GetArrayFromImage(image)

        # apply normalization to the image
        intensity_scaler = ScaleIntensityRanged(
            keys=["img"],
            a_min=-1350.0, 
            a_max=150.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
            )
        data = {"img": image_array}
        image_rescaled_array = intensity_scaler(data)
        image_rescaled = sitk.GetImageFromArray(image_rescaled_array['img'])
        image_rescaled = sitk.Cast(image_rescaled, sitk.sitkFloat32)

        source_mask_file = path_base.joinpath(f"Data/{row['mask_path']}")

        parts_name = str(source_mask_file).split(os.sep)
        dataset_name = parts_name[-3]
        if 'LIDC' in dataset_name:
            image_new_name = f"LIDC_{count:04d}_{0:04d}.nii.gz"
            mask_new_name = f"LIDC_{count:04d}.nii.gz"
        elif '626752' in str(dataset_name):
            image_new_name = f"PAP_{count:04d}_{0:04d}.nii.gz"
            mask_new_name = f"PAP_{count:04d}.nii.gz"
        elif 'ANON' in str(dataset_name):
            image_new_name = f"PAGNI_{count:04d}_{0:04d}.nii.gz"
            mask_new_name = f"PAGNI_{count:04d}.nii.gz"
        elif 'Sotiria' in str(dataset_name):
            image_new_name = f"SOTIRIA_{count:04d}_{0:04d}.nii.gz"
            mask_new_name = f"SOTIRIA_{count:04d}.nii.gz"

        mapping['dataset'].append(dataset_name)
        mapping['nodule'].append(parts_name[-2])
        mapping['mask'].append(parts_name[-1].replace('.nii.gz', ''))
        mapping['image_new_name'].append(image_new_name)
        mapping['mask_new_name'].append(mask_new_name)
        print(f"Saving {dataset_name}_{parts_name[-2]} as {parts_name[-1]} with new name image:{image_new_name}, mask:{mask_new_name}!")

        sitk.WriteImage(image_rescaled, path_images.joinpath(image_new_name))
        if path_labels is not None:
            shutil.copy2(source_mask_file, path_labels.joinpath(mask_new_name))
        count += 1

    return mapping, count

def create_split_json(path_base: Path):
    preprocessed_dir = Path(path_base).joinpath("nnUnet_preprocessed/Dataset001_LungNodule")

    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    train_mapping_path = Path("/home/kdovrou/PhD/Data/train_mapping.csv")
    val_mapping_path = Path("/home/kdovrou/PhD/Data/val_mapping.csv")

    train = pd.read_csv(train_mapping_path)
    val = pd.read_csv(val_mapping_path)

    train_cases = train['mask_new_name'].tolist()
    val_cases = val['mask_new_name'].tolist()

    train_cases = [case.replace(".nii.gz", "") for case in train_cases]
    val_cases = [case.replace(".nii.gz", "") for case in val_cases]

    splits = [{
        "train": train_cases,
        "val": val_cases
    }]

    with open(Path(preprocessed_dir).joinpath("splits_final.json"), "w") as f:
        json.dump(splits, f, indent=4)

if __name__ == "__main__":
    # create the directories

    path_base = Path('/home/kdovrou/PhD')

    path_dataset = path_base.joinpath('models/nnUnet_raw/Dataset001_LungNodule')
    path_dataset.mkdir(exist_ok=True, parents=True)

    path_imagesTr = path_dataset.joinpath('imagesTr') 
    path_imagesTr.mkdir(exist_ok=True, parents=True)
    path_labelsTr = path_dataset.joinpath('labelsTr')
    path_labelsTr.mkdir(exist_ok=True, parents=True)
    path_imagesTs = path_dataset.joinpath('imagesTs')
    path_imagesTs.mkdir(exist_ok=True, parents=True)

    # read splits
    path_train = path_base.joinpath('Data/train.csv')
    path_val = path_base.joinpath('Data/val.csv')
    path_test = path_base.joinpath('Data/test.csv')

    train_cases = pd.read_csv(path_train)
    val_cases = pd.read_csv(path_val)
    test_cases = pd.read_csv(path_test)

    # # convert data (update names and copy data to the new location)
    # count = 0 
    # train_mapping, count = convert_data(train_cases, count, path_imagesTr, path_labelsTr)
    # val_mapping, count = convert_data(val_cases, count, path_imagesTr, path_labelsTr)
    # test_mapping, count = convert_data(test_cases, count, path_imagesTs)

    # # save the mapping to a CSV file
    # pd.DataFrame(train_mapping).to_csv(path_base.joinpath('Data/train_mapping.csv'), index=False)
    # pd.DataFrame(val_mapping).to_csv(path_base.joinpath('Data/val_mapping.csv'), index=False)
    # pd.DataFrame(test_mapping).to_csv(path_base.joinpath('Data/test_mapping.csv'), index=False)


    generate_dataset_json(path_dataset,
                          channel_names={0: 'CT'},
                          labels={
                              'background': 0,
                              'lung_nodule': 1
                          },
                          num_training_cases=len(train_cases) + len(val_cases),
                          file_ending='.nii.gz',
                          licence='MIT',
                          converted_by='KaterinaDovrou'
                          )
    
    create_split_json(path_base.joinpath('models'))

