import os
import multiprocessing
import numpy as np
import torch
import SimpleITK as sitk
import nibabel
import argparse

from copy import deepcopy
from math import pi
from time import sleep
from typing import Tuple, Union, List, Optional
from torch import nn
from torch._dynamo import OptimizedModule
from tqdm import tqdm

from monai.transforms import (
    RandRotated,
    RandZoomd,
    RandFlipd,
    RandGaussianNoised,
    RandAdjustContrastd,
    Invertd,
    InvertibleTransform,
)
from monai.data import MetaTensor

# Ensure compiled models are disabled
os.environ['nnUNet_compile'] = 'False'

from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.imageio.nibabel_reader_writer import NibabelIO, NibabelIOWithReorient
from nnunetv2.inference.predict_from_raw_data import (
    nnUNetPredictor,
    check_workers_alive_and_busy,
    compute_gaussian,
    _getDefaultValue
)
from nnunetv2.inference.export_prediction import (
    export_prediction_from_logits,
    convert_predicted_logits_to_segmentation_with_correct_shape
)
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import load_json, join, maybe_mkdir_p, isdir, save_json
from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot


def get_tta_transforms():
    """
    Define a list of 10 TTA transforms applied to the input volume using MONAI:
    1. 'original': Identity / no transformation.
    2. RandRotated(keys=["img"], prob=1, range_x=5*pi/180, range_y=5*pi/180, range_z=10*pi/180, mode="bilinear", padding_mode="border", lazy=False)
    3. RandRotated(keys=["img"], prob=1, range_x=3*pi/180, range_y=3*pi/180, range_z=5*pi/180, mode="bilinear", padding_mode="border", lazy=False)
    4. RandZoomd(keys=["img"], prob=1, min_zoom=0.9, max_zoom=1.1, mode="bilinear", padding_mode="edge", lazy=False)
    5. RandZoomd(keys=["img"], prob=1, min_zoom=0.95, max_zoom=1.05, mode="bilinear", padding_mode="edge", lazy=False)
    6. RandFlipd(keys=["img"], prob=1, spatial_axis=2)
    7. RandGaussianNoised(keys=["img"], prob=1, std=0.01)
    8. RandGaussianNoised(keys=["img"], prob=1, std=0.03)
    9. RandAdjustContrastd(keys=["img"], prob=1, gamma=(0.9, 1.1))
    10. RandAdjustContrastd(keys=["img"], prob=1, gamma=(0.95, 1.05))
    """
    return [
        "original",
        RandRotated(keys=["img"], prob=1, range_x=5 * pi / 180, range_y=5 * pi / 180, range_z=10 * pi / 180, mode="bilinear", padding_mode="border", lazy=False),
        RandRotated(keys=["img"], prob=1, range_x=3 * pi / 180, range_y=3 * pi / 180, range_z=5 * pi / 180, mode="bilinear", padding_mode="border", lazy=False),
        RandZoomd(keys=["img"], prob=1, min_zoom=0.9, max_zoom=1.1, mode="bilinear", padding_mode="edge", lazy=False),
        RandZoomd(keys=["img"], prob=1, min_zoom=0.95, max_zoom=1.05, mode="bilinear", padding_mode="edge", lazy=False),
        RandFlipd(keys=["img"], prob=1, spatial_axis=2),
        RandGaussianNoised(keys=["img"], prob=1, std=0.01),
        RandGaussianNoised(keys=["img"], prob=1, std=0.03),
        RandAdjustContrastd(keys=["img"], prob=1, gamma=(0.9, 1.1)),
        RandAdjustContrastd(keys=["img"], prob=1, gamma=(0.95, 1.05))
    ]


def write_float_image(data: np.ndarray, output_fname: str, properties: dict, rw) -> None:
    if isinstance(rw, SimpleITKIO):
        assert data.ndim == 3, 'data must be 3d'
        output_dimension = len(properties['sitk_stuff']['spacing'])
        assert 1 < output_dimension < 4
        if output_dimension == 2:
            data = data[0]
        itk_image = sitk.GetImageFromArray(data.astype(np.float32, copy=False))
        itk_image.SetSpacing(properties['sitk_stuff']['spacing'])
        itk_image.SetOrigin(properties['sitk_stuff']['origin'])
        itk_image.SetDirection(properties['sitk_stuff']['direction'])
        sitk.WriteImage(itk_image, output_fname, True)
    elif isinstance(rw, (NibabelIO, NibabelIOWithReorient)):
        # Nibabel transposes the axes (2, 1, 0)
        data_to_write = data.transpose((2, 1, 0)).astype(np.float32, copy=False)
        seg_nib = nibabel.Nifti1Image(data_to_write, affine=properties['nibabel_stuff']['original_affine'])
        nibabel.save(seg_nib, output_fname)
    else:
        # Generic fallback
        try:
            assert data.ndim == 3, 'data must be 3d'
            itk_image = sitk.GetImageFromArray(data.astype(np.float32, copy=False))
            sitk.WriteImage(itk_image, output_fname, True)
        except Exception:
            rw.write_seg(data, output_fname, properties)


def export_uncertainty(uncertainty: np.ndarray,
                       properties_dict: dict,
                       configuration_manager: ConfigurationManager,
                       plans_manager: PlansManager,
                       output_file: str):
    # resample to original shape (after cropping)
    spacing_transposed = [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    current_spacing = configuration_manager.spacing if \
        len(configuration_manager.spacing) == \
        len(properties_dict['shape_after_cropping_and_before_resampling']) else \
        [spacing_transposed[0], *configuration_manager.spacing]

    # Resampling function expects (C, X, Y, Z). Let's expand uncertainty to have a channel dim if it's 3D.
    if uncertainty.ndim == 3:
        uncertainty = uncertainty[None]

    resampled_uncertainty = configuration_manager.resampling_fn_probabilities(
        uncertainty,
        properties_dict['shape_after_cropping_and_before_resampling'],
        current_spacing,
        [properties_dict['spacing'][i] for i in plans_manager.transpose_forward]
    )

    # Revert cropping
    uncertainty_reverted_cropping = np.zeros((1, *properties_dict['shape_before_cropping']), dtype=np.float32)
    uncertainty_reverted_cropping = insert_crop_into_image(
        uncertainty_reverted_cropping,
        resampled_uncertainty,
        properties_dict['bbox_used_for_cropping']
    )

    if isinstance(uncertainty_reverted_cropping, torch.Tensor):
        uncertainty_reverted_cropping = uncertainty_reverted_cropping.cpu().numpy()

    # Revert transpose
    uncertainty_reverted_cropping = uncertainty_reverted_cropping.transpose([0] + [i + 1 for i in plans_manager.transpose_backward])

    # Squeeze the channel dimension to make it 3D again
    uncertainty_reverted_cropping = np.squeeze(uncertainty_reverted_cropping, axis=0)

    # Save
    rw = plans_manager.image_reader_writer_class()
    write_float_image(uncertainty_reverted_cropping, output_file, properties_dict, rw)


class nnUNetPredictorTTA(nnUNetPredictor):
    def __init__(self,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True,
                 save_uncertainty: bool = True,
                 uncertainty_metrics: List[str] = ['variance', 'entropy']):
        super().__init__(tile_step_size=tile_step_size,
                         use_gaussian=use_gaussian,
                         use_mirroring=use_mirroring,
                         perform_everything_on_device=perform_everything_on_device,
                         device=device,
                         verbose=verbose,
                         verbose_preprocessing=verbose_preprocessing,
                         allow_tqdm=allow_tqdm)
        self.save_uncertainty = save_uncertainty
        self.uncertainty_metrics = uncertainty_metrics

    def predict_logits_single_pass(self, data: torch.Tensor) -> torch.Tensor:
        """
        Runs sliding window inference on data, ensembling across all loaded model parameters (folds).
        """
        prediction = None
        for params in self.list_of_parameters:
            if not isinstance(self.network, OptimizedModule):
                self.network.load_state_dict(params)
            else:
                self.network._orig_mod.load_state_dict(params)

            logits = self.predict_sliding_window_return_logits(data)
            if prediction is None:
                prediction = logits
            else:
                prediction += logits

        if len(self.list_of_parameters) > 1:
            prediction /= len(self.list_of_parameters)

        return prediction

    def predict_tta_logits_and_uncertainties(self, data: torch.Tensor):
        self.network = self.network.to(self.device)
        self.network.eval()

        # 1. Ensure the input volume is shaped as a 4D tensor (C, W, H, D)
        volume = data
        if volume.ndim == 3:
            volume = volume.unsqueeze(0)
        elif volume.ndim == 5 and volume.shape[0] == 1:
            volume = volume.squeeze(0)
        assert volume.ndim == 4, f"Input volume must be 4D (C, W, H, D), got {volume.shape}"

        transforms = get_tta_transforms()
        print(f"Running Test-Time Augmentation (TTA) with {len(transforms)} passes...")

        all_probs = []
        for pass_idx, t in enumerate(transforms):
            transform_name = "original" if t == "original" else t.__class__.__name__
            if self.verbose:
                print(f"TTA Pass {pass_idx + 1}/{len(transforms)}: {transform_name}")

            # 1. Forward Pass and Data Augmentation
            vol_meta = MetaTensor(volume.clone())
            aug_data = {"img": vol_meta} if t == "original" else t({"img": vol_meta})
            aug_img = aug_data["img"]

            aug_img_tensor = aug_img if isinstance(aug_img, torch.Tensor) else torch.as_tensor(aug_img)
            logits = self.predict_logits_single_pass(aug_img_tensor)

            # Continuous probabilities
            probs = self.label_manager.apply_inference_nonlin(logits)

            # 2. Invert Spatial Transformations
            if isinstance(t, InvertibleTransform):
                pred_tensor = probs.to(aug_img.device)
                aug_data["pred"] = MetaTensor(
                    pred_tensor,
                    meta=deepcopy(aug_img.meta) if hasattr(aug_img, "meta") else {},
                    applied_operations=deepcopy(aug_img.applied_operations) if hasattr(aug_img, "applied_operations") else []
                )

                inverter = Invertd(
                    keys="pred",
                    transform=t,
                    orig_keys="img",
                    meta_keys="pred_meta_dict",
                    orig_meta_keys="img_meta_dict",
                    nearest_interp=False  # Keep continuous probabilities, do not discretize yet
                )
                inverted_data = inverter(aug_data)
                aligned_prob = inverted_data["pred"]
            else:
                aug_data["pred"] = probs
                aligned_prob = aug_data["pred"]

            # Store on CPU to avoid accumulating VRAM
            aligned_prob_tensor = torch.as_tensor(aligned_prob).detach().to("cpu", dtype=torch.float32)
            all_probs.append(aligned_prob_tensor)

        # 3. Ensemble Aggregation & Uncertainty Estimation
        # Stack all passes: shape (N_TTA, C, W, H, D)
        all_probs = torch.stack(all_probs)
        if all_probs.ndim == 6 and all_probs.shape[1] == 1:
            all_probs = all_probs.squeeze(1)

        # 1. Mean Probabilities
        mean_probs = torch.mean(all_probs, dim=0)

        # Convert back to pseudo-logits for nnUNet export pipeline
        if self.label_manager.has_regions:
            mean_logits = torch.logit(torch.clamp(mean_probs, 1e-7, 1.0 - 1e-7))
        else:
            mean_logits = torch.log(mean_probs + 1e-8)

        # 2. Variance Uncertainty Map: variance across passes, then mean across channels
        var_per_channel = torch.var(all_probs, dim=0, unbiased=False)
        variance_map = torch.mean(var_per_channel, dim=0)
        while variance_map.ndim > 3:
            variance_map = torch.mean(variance_map, dim=0)

        # 3. Entropy Uncertainty Map (Shannon entropy across classes)
        entropy_map = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=0)
        while entropy_map.ndim > 3:
            entropy_map = torch.sum(entropy_map, dim=0)

        # 4. Pairwise DSC Diversity
        num_total_passes = all_probs.shape[0]
        dsc_dict = {'pairwise_dscs': [], 'mean_dscs_per_class': [], 'mean_dsc_overall': 0.0}
        foreground_classes = self.label_manager.foreground_regions if self.label_manager.has_regions else self.label_manager.foreground_labels

        all_segs = []
        for probs_pass in all_probs:
            if self.label_manager.has_regions:
                seg = torch.zeros(probs_pass.shape[1:], dtype=torch.int16, device=probs_pass.device)
                for i, c in enumerate(self.label_manager.regions_class_order):
                    seg[probs_pass[i] > 0.5] = c
            else:
                seg = probs_pass.argmax(0)
            all_segs.append(seg)
        all_segs = torch.stack(all_segs)

        if num_total_passes > 1 and len(foreground_classes) > 0:
            all_pairwise_dscs = []
            for i in range(num_total_passes):
                for j in range(i + 1, num_total_passes):
                    dscs_per_class = []
                    for c in foreground_classes:
                        if self.label_manager.has_regions:
                            c_idx = self.label_manager.regions_class_order.index(c) if c in self.label_manager.regions_class_order else c
                            seg_i = (all_probs[i][c_idx] > 0.5)
                            seg_j = (all_probs[j][c_idx] > 0.5)
                        elif all_probs.shape[1] == 1:
                            seg_i = (all_probs[i][0] > 0.5)
                            seg_j = (all_probs[j][0] > 0.5)
                        elif len(foreground_classes) == 1 and all_probs.shape[1] == 2:
                            seg_i = (all_probs[i][c] > 0.5)
                            seg_j = (all_probs[j][c] > 0.5)
                        else:
                            seg_i = (all_segs[i] == c)
                            seg_j = (all_segs[j] == c)

                        intersection = (seg_i & seg_j).sum().float()
                        union = seg_i.sum().float() + seg_j.sum().float()
                        if union > 0:
                            dsc = (2.0 * intersection) / union
                        else:
                            dsc = torch.tensor(1.0, device=all_probs.device)
                        dscs_per_class.append(dsc.item())
                    all_pairwise_dscs.append(dscs_per_class)

            dsc_dict['pairwise_dscs'] = all_pairwise_dscs
            mean_dscs_per_class = np.mean(all_pairwise_dscs, axis=0).tolist()
            dsc_dict['mean_dscs_per_class'] = mean_dscs_per_class
            dsc_dict['mean_dsc_overall'] = float(np.mean(mean_dscs_per_class))

        return mean_logits, variance_map, entropy_map, dsc_dict

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = 3,
                                   folder_for_probabilities: str = None):
        """
        Modified to compute and export uncertainty maps during TTA prediction.
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            r_unc = []

            for preprocessed in data_iterator:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)} with TTA:')
                else:
                    print(f'\nPredicting image of shape {data.shape} with TTA:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')
                properties = preprocessed['data_properties']

                # Avoid swamping disk
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                # Run TTA prediction to get logits, variance, and entropy
                mean_logits, variance_map, entropy_map, dsc_dict = self.predict_tta_logits_and_uncertainties(data)
                prediction = mean_logits.cpu().detach().numpy()

                if ofile is not None:
                    # Save pairwise DSCs to JSON: {case}_tta_dsc.json
                    dsc_ofile = ofile + '_tta_dsc.json'
                    print(f'Saving pairwise DSCs to {dsc_ofile}')
                    save_json(dsc_dict, dsc_ofile)

                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.apply_async(
                            export_prediction_from_logits,
                            (prediction, properties, self.configuration_manager, self.plans_manager,
                             self.dataset_json, ofile, save_probabilities, folder_for_probabilities)
                        )
                    )

                    if self.save_uncertainty:
                        for metric in self.uncertainty_metrics:
                            metric_map = variance_map if metric == 'variance' else entropy_map
                            metric_np = metric_map.cpu().detach().numpy()
                            suffix = f'_uncertainty_{metric}'
                            uncertainty_ofile = ofile + suffix + self.dataset_json['file_ending']
                            print(f'sending off {metric} uncertainty to background worker for resampling and export')
                            r_unc.append(
                                export_pool.apply_async(
                                    export_uncertainty,
                                    (metric_np, properties, self.configuration_manager, self.plans_manager,
                                     uncertainty_ofile)
                                )
                            )
                else:
                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.apply_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape,
                            (prediction, self.plans_manager,
                             self.configuration_manager, self.label_manager,
                             properties,
                             save_probabilities)
                        )
                    )

                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')

            print("GPU prediction completed. Waiting for remaining segmentation and uncertainty exports to finish...")
            all_results = r + r_unc
            ret = [None] * len(r)
            with tqdm(desc="Collecting results", total=len(all_results),
                      disable=not self.allow_tqdm) as pbar:
                for i, result in enumerate(r):
                    while True:
                        all_alive = all([j.is_alive() for j in worker_list])
                        if not all_alive:
                            raise RuntimeError('Segmentation export worker died. It was likely killed by '
                                               'your OS because of insufficient available CPU RAM.')
                        try:
                            ret[i] = result.get(timeout=0.1)
                            break
                        except multiprocessing.TimeoutError:
                            pass
                    pbar.update()
                for result in r_unc:
                    while True:
                        all_alive = all([j.is_alive() for j in worker_list])
                        if not all_alive:
                            raise RuntimeError('Segmentation export worker died. It was likely killed by '
                                               'your OS because of insufficient available CPU RAM.')
                        try:
                            result.get(timeout=0.1)
                            break
                        except multiprocessing.TimeoutError:
                            pass
                    pbar.update()
            print("Segmentation and uncertainty export complete.")

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        compute_gaussian.cache_clear()
        empty_cache(self.device)
        return ret

    def predict_from_files_sequential(self,
                                      list_of_lists_or_source_folder: Union[str, List[List[str]]],
                                      output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                                      save_probabilities: bool = False,
                                      overwrite: bool = True,
                                      folder_with_segs_from_prev_stage: str = None,
                                      folder_for_probabilities: str = None):
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, 0, 1,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)

        if output_filename_truncated is None:
            output_filename_truncated = [None] * len(list_of_lists_or_source_folder)
        if seg_from_prev_stage_files is None:
            seg_from_prev_stage_files = [None] * len(list_of_lists_or_source_folder)

        ret = []
        for li, of, sps in zip(list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files):
            data, seg, data_properties = preprocessor.run_case(
                li,
                sps,
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json
            )
            if folder_with_segs_from_prev_stage is not None:
                seg_onehot = convert_labelmap_to_one_hot(seg[0], label_manager.foreground_labels, data.dtype)
                data = np.vstack((data, seg_onehot))

            print(f'perform_everything_on_device: {self.perform_everything_on_device}')

            mean_logits, variance_map, entropy_map, dsc_dict = self.predict_tta_logits_and_uncertainties(torch.from_numpy(data))
            prediction = mean_logits.cpu().detach().numpy()

            if of is not None:
                # Save pairwise DSCs to JSON: {case}_tta_dsc.json
                dsc_ofile = of + '_tta_dsc.json'
                print(f'Saving pairwise DSCs to {dsc_ofile}')
                save_json(dsc_dict, dsc_ofile)

                export_prediction_from_logits(prediction, data_properties, self.configuration_manager, self.plans_manager,
                                              self.dataset_json, of, save_probabilities, folder_for_probabilities)

                if self.save_uncertainty:
                    for metric in self.uncertainty_metrics:
                        metric_map = variance_map if metric == 'variance' else entropy_map
                        metric_np = metric_map.cpu().detach().numpy()
                        suffix = f'_uncertainty_{metric}'
                        uncertainty_ofile = of + suffix + self.dataset_json['file_ending']
                        print(f'Saving {metric} uncertainty map sequentially...')
                        export_uncertainty(metric_np, data_properties, self.configuration_manager, self.plans_manager,
                                           uncertainty_ofile)
            else:
                ret.append(convert_predicted_logits_to_segmentation_with_correct_shape(prediction, self.plans_manager,
                                                                                      self.configuration_manager, self.label_manager,
                                                                                      data_properties,
                                                                                      save_probabilities))

        compute_gaussian.cache_clear()
        empty_cache(self.device)
        return ret


def predict_entry_point():
    parser = argparse.ArgumentParser(description='Use this to run Test-Time Augmentation (TTA) inference with nnU-Net.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable inner sliding window mirroring data augmentation. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('-prob_folder', type=str, required=False, default=None,
                            help='Folder where probability maps will be saved. Must be used with --save_probabilities.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-npp', type=int, required=False, default=_getDefaultValue('nnUNet_npp', int, 3),
                        help='Number of processes used for preprocessing. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-nps', type=int, required=False, default=_getDefaultValue('nnUNet_nps', int, 3),
                        help='Number of processes used for segmentation export. More is not always better. Beware of '
                             'out-of-RAM issues. Default: 3')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict calls that you will be making. Default: 1 (= this one '
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with '
                             'num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts '
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')
    parser.add_argument('--not_on_device', action='store_true', required=False, default=False,
                        help="Set this flag to disable perform_everything_on_device. Recommended for large cases that "
                             "occupy more VRAM than available")

    # Test-Time Augmentation (TTA) Specific Arguments
    parser.add_argument('--disable_uncertainty', action='store_true', required=False, default=False,
                        help='Set this flag to disable saving uncertainty maps.')
    parser.add_argument('-uncertainty_metrics', nargs='+', type=str, choices=['variance', 'entropy'], default=['variance', 'entropy'],
                        help='Uncertainty metrics to calculate and save. Default: variance entropy')

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    if not isdir(args.o):
        maybe_mkdir_p(args.o)

    assert args.part_id < args.num_parts, 'Do you even read the documentation? See nnUNetv2_predict -h.'
    assert args.device in ['cpu', 'cuda', 'mps'], f'-device must be either cpu, mps or cuda. Got: {args.device}.'

    if args.device == 'cpu':
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictorTTA(tile_step_size=args.step_size,
                                   use_gaussian=True,
                                   use_mirroring=not args.disable_tta,
                                   perform_everything_on_device=not args.not_on_device,
                                   device=device,
                                   verbose=args.verbose,
                                   verbose_preprocessing=args.verbose,
                                   allow_tqdm=not args.disable_progress_bar,
                                   save_uncertainty=not args.disable_uncertainty,
                                   uncertainty_metrics=args.uncertainty_metrics)

    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )

    run_sequential = args.nps == 0 and args.npp == 0

    if run_sequential:
        print("Running in non-multiprocessing mode")
        predictor.predict_from_files_sequential(args.i, args.o, save_probabilities=args.save_probabilities,
                                                overwrite=not args.continue_prediction,
                                                folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                                folder_for_probabilities=args.prob_folder)
    else:
        predictor.predict_from_files(args.i, args.o, save_probabilities=args.save_probabilities,
                                     overwrite=not args.continue_prediction,
                                     num_processes_preprocessing=args.npp,
                                     num_processes_segmentation_export=args.nps,
                                     folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                     num_parts=args.num_parts,
                                     part_id=args.part_id,
                                     folder_for_probabilities=args.prob_folder)


if __name__ == '__main__':
    predict_entry_point()

