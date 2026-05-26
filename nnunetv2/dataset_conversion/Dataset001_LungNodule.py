import os
from pathlib import Path
from monai.transforms import ScaleIntensityRanged
import pandas as pd
import SimpleITK as sitk
import shutil

def convert_data(data: pd.DataFrame, count:int, path_images: Path | str, path_labels: Path | str=None):
    mapping = {
        'dataset': [],
        'nodule': [],
        'mask': [],
        'new_name': []
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
            new_name = f"LIDC_{count:04d}.nii.gz"
        elif '626752' in str(dataset_name):
            new_name = f"PAP_{count:04d}.nii.gz"
        elif 'ANON' in str(dataset_name):
            new_name = f"PAGNI_{count:04d}.nii.gz"
        elif 'Sotiria' in str(dataset_name):
            new_name = f"SOTIRIA_{count:04d}.nii.gz"

        mapping['dataset'].append(dataset_name)
        mapping['nodule'].append(parts_name[-2])
        mapping['mask'].append(parts_name[-1].replace('.nii.gz', ''))
        mapping['new_name'].append(new_name)
        print(f"Saving {dataset_name}_{parts_name[-2]} as {parts_name[-1]} with new name {new_name}!")

        sitk.WriteImage(image_rescaled, path_images.joinpath(new_name))
        if path_labels is not None:
            shutil.copy2(source_mask_file, path_labels.joinpath(new_name))
        count += 1

    return mapping, count

if __name__ == "__main__":
    # create the directories

    path_base = Path('/home/kdovrou/PhD')

    path_dataset = path_base.joinpath('models/dataset/nnUnet_raw/Dataset001_LungNodule')
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

    # convert data (update names and copy data to the new location)
    count = 0 
    train_mapping, count = convert_data(train_cases, count, path_imagesTr, path_labelsTr)
    val_mapping, count = convert_data(val_cases, count, path_imagesTr, path_labelsTr)
    test_mapping, count = convert_data(test_cases, count, path_imagesTs)

    # save the mapping to a CSV file
    pd.DataFrame(train_mapping).to_csv(path_base.joinpath('Data/train_mapping.csv'), index=False)
    pd.DataFrame(val_mapping).to_csv(path_base.joinpath('Data/val_mapping.csv'), index=False)
    pd.DataFrame(test_mapping).to_csv(path_base.joinpath('Data/test_mapping.csv'), index=False)

