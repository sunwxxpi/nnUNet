import argparse
import gc
import os
from pathlib import Path
from typing import Union, Tuple

import nnunetv2
import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from batchgenerators.utilities.file_and_folder_operations import load_json, join
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from tqdm import tqdm


class CustomPredictor(nnUNetPredictor):
    """
    메모리 사용량을 최대한 줄인 nnUNetPredictor 구현
    """

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_folds: Union[Tuple[Union[int, str]], None],
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        학습된 모델 폴더로부터 예측에 필요한 환경을 초기화합니다.
        """
        if use_folds is None:
            use_folds = self.auto_detect_available_folds(model_training_output_dir, checkpoint_name)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if isinstance(use_folds, str):
            use_folds = [use_folds]

        parameters = []
        for i, f in enumerate(use_folds):
            f = int(f) if f != 'all' else f
            checkpoint_path = join(model_training_output_dir, f'fold_{f}', checkpoint_name)
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=False)
            if i == 0:
                trainer_name = checkpoint['trainer_name']
                configuration_name = checkpoint['init_args']['configuration']
                inference_allowed_mirroring_axes = checkpoint.get('inference_allowed_mirroring_axes')
            parameters.append(checkpoint_path)
            del checkpoint  # 즉시 메모리 해제
            gc.collect()

        configuration_manager = plans_manager.get_configuration(configuration_name)
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.list_of_parameters = parameters
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        self.label_manager = plans_manager.get_label_manager(dataset_json)

    @torch.inference_mode()
    def _predict_streamed_ultra_efficient(self, data: torch.Tensor, props: dict) -> np.ndarray:
        """
        극도로 메모리 효율적인 스트리밍 예측 메서드.
        주요 개선사항:
        1. 큰 logits_accumulator_cpu 대신 직접 voting 방식 사용
        2. float16 대신 uint8 카운터 사용
        3. 패치별 즉시 처리로 메모리 최소화
        """
        # 1. 최종 분할 결과를 저장할 공간
        final_segmentation = np.zeros(props['shape_before_cropping'], dtype=np.uint8)
        slicer_cropping = bounding_box_to_slice(props['bbox_used_for_cropping'])

        # 2. 데이터 패딩 및 슬라이더 생성
        data, slicer_revert_padding = pad_nd_image(
            data, self.configuration_manager.patch_size, 'constant', {'value': 0}, True, None
        )
        slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

        # 3. 가우시안 필터 생성
        gaussian = compute_gaussian(
            tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
            value_scaling_factor=10, device=self.device, dtype=torch.float16
        )

        num_classes = self.label_manager.num_segmentation_heads
        
        # 4. ★ 핵심 최적화: voting 카운터 사용 (float16 logits 대신)
        # 각 복셀에서 각 클래스가 몇 번 선택되었는지 카운트 (uint8로 매우 가벼움)
        vote_counter = np.zeros((num_classes, *data.shape[1:]), dtype=np.uint8)
        
        # 5. 각 fold별로 순차 처리
        for fold_idx, p in enumerate(self.list_of_parameters):
            print(f"Processing fold {fold_idx + 1}/{len(self.list_of_parameters)}")
            
            # 모델 로드
            checkpoint = torch.load(p, map_location=torch.device('cpu'), weights_only=False)
            self.network.load_state_dict(checkpoint['network_weights'])
            del checkpoint
            gc.collect()
            
            self.network.to(self.device)
            self.network.eval()
            
            # 슬라이딩 윈도우 방식으로 예측
            with torch.autocast(self.device.type, enabled=True):
                for sl in tqdm(slicers, disable=not self.allow_tqdm, desc=f"Fold {fold_idx + 1}"):
                    # 패치 예측
                    patch_data = data[sl][None].to(self.device)
                    predicted_patch = self._internal_maybe_mirror_and_predict(patch_data)[0]
                    
                    # ★ 핵심: 가중치 적용 후 즉시 argmax로 클래스 선택
                    weighted_patch = predicted_patch * gaussian
                    predicted_classes = torch.argmax(weighted_patch, dim=0).cpu().numpy().astype(np.uint8)
                    del predicted_patch, weighted_patch  # 즉시 메모리 해제
                    
                    # 선택된 클래스에 대해 voting 카운터 증가
                    for class_idx in range(num_classes):
                        class_mask = (predicted_classes == class_idx)
                        vote_counter[class_idx][sl[1:]][class_mask] += 1
                    
                    del predicted_classes, patch_data
                    
                    # 주기적 메모리 정리
                    if len([s for s in slicers if s == sl]) % 10 == 0:
                        torch.cuda.empty_cache()
            
            # fold 처리 완료 후 네트워크를 CPU로 이동하여 VRAM 절약
            self.network.to('cpu')
            gc.collect()

        # 6. voting 결과를 기반으로 최종 segmentation 결정
        print("Computing final segmentation from votes...")
        final_classes = np.argmax(vote_counter, axis=0).astype(np.uint8)
        del vote_counter  # 큰 메모리 즉시 해제
        gc.collect()

        # 패딩 제거 - 차원 수정
        if len(slicer_revert_padding) > len(final_classes.shape):
            # 4D slicer를 3D로 변환 (첫 번째 채널 차원 제거)
            slicer_revert_padding_3d = slicer_revert_padding[1:]
        else:
            slicer_revert_padding_3d = slicer_revert_padding
            
        seg_result_cropped = final_classes[slicer_revert_padding_3d]
        del final_classes
        gc.collect()

        # 원본 이미지 크기로 복원
        final_segmentation[slicer_cropping] = seg_result_cropped
        final_segmentation = final_segmentation.transpose(self.plans_manager.transpose_backward)

        return final_segmentation

    @torch.inference_mode()
    def _predict_streamed_memory_mapped(self, data: torch.Tensor, props: dict) -> np.ndarray:
        """
        대안: 메모리 매핑을 사용한 버전 (더 극단적인 메모리 절약)
        """
        import tempfile
        
        final_segmentation = np.zeros(props['shape_before_cropping'], dtype=np.uint8)
        slicer_cropping = bounding_box_to_slice(props['bbox_used_for_cropping'])

        data, slicer_revert_padding = pad_nd_image(
            data, self.configuration_manager.patch_size, 'constant', {'value': 0}, True, None
        )
        slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

        gaussian = compute_gaussian(
            tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
            value_scaling_factor=10, device=self.device, dtype=torch.float16
        )

        num_classes = self.label_manager.num_segmentation_heads
        
        # 임시 파일에 메모리 매핑된 배열 생성 (디스크 기반)
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        vote_counter = np.memmap(temp_file.name, dtype=np.uint8, mode='w+', 
                               shape=(num_classes, *data.shape[1:]))
        
        try:
            # fold별 처리 (이전과 동일)
            for fold_idx, p in enumerate(self.list_of_parameters):
                checkpoint = torch.load(p, map_location=torch.device('cpu'), weights_only=False)
                self.network.load_state_dict(checkpoint['network_weights'])
                del checkpoint
                gc.collect()
                
                self.network.to(self.device)
                self.network.eval()
                
                with torch.autocast(self.device.type, enabled=True):
                    for sl in tqdm(slicers, disable=not self.allow_tqdm, desc=f"Fold {fold_idx + 1}"):
                        patch_data = data[sl][None].to(self.device)
                        predicted_patch = self._internal_maybe_mirror_and_predict(patch_data)[0]
                        
                        weighted_patch = predicted_patch * gaussian
                        predicted_classes = torch.argmax(weighted_patch, dim=0).cpu().numpy().astype(np.uint8)
                        del predicted_patch, weighted_patch, patch_data
                        
                        for class_idx in range(num_classes):
                            class_mask = (predicted_classes == class_idx)
                            vote_counter[class_idx][sl[1:]][class_mask] += 1
                        
                        del predicted_classes
                
                self.network.to('cpu')
                gc.collect()

            # 최종 결과 계산
            final_classes = np.argmax(vote_counter, axis=0).astype(np.uint8)
            
        finally:
            # 메모리 매핑 파일 정리
            del vote_counter
            os.unlink(temp_file.name)

        # 패딩 제거 - 차원 수정
        if len(slicer_revert_padding) > len(final_classes.shape):
            # 4D slicer를 3D로 변환 (첫 번째 채널 차원 제거)
            slicer_revert_padding_3d = slicer_revert_padding[1:]
        else:
            slicer_revert_padding_3d = slicer_revert_padding
            
        seg_result_cropped = final_classes[slicer_revert_padding_3d]
        del final_classes
        gc.collect()

        final_segmentation[slicer_cropping] = seg_result_cropped
        final_segmentation = final_segmentation.transpose(self.plans_manager.transpose_backward)

        return final_segmentation

    def predict_single_npy_array(self, input_image: np.ndarray, image_properties: dict,
                                 segmentation_previous_stage: np.ndarray = None,
                                 use_memory_mapping: bool = False):
        """
        예측 메인 메서드
        use_memory_mapping: True면 디스크 기반 메모리 매핑 사용 (더 느리지만 RAM 사용량 최소)
        """
        torch.set_num_threads(os.cpu_count())

        # 전처리
        preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)
        data, _, props = preprocessor.run_case_npy(input_image, None, image_properties,
                                              self.plans_manager, self.configuration_manager,
                                              self.dataset_json)
        del input_image
        gc.collect()

        # 예측 실행
        data = torch.from_numpy(data)
        if use_memory_mapping:
            segmentation = self._predict_streamed_memory_mapped(data, props)
        else:
            segmentation = self._predict_streamed_ultra_efficient(data, props)
        
        del data
        gc.collect()

        return segmentation


def predict_semseg(im, prop, semseg_trained_model, semseg_folds, use_memory_mapping=False):
    """
    예측 헬퍼 함수
    """
    pred_semseg = CustomPredictor(
        tile_step_size=0.5,
        use_mirroring=True,
        use_gaussian=True,
        perform_everything_on_device=False,
        allow_tqdm=True
    )
    pred_semseg.initialize_from_trained_model_folder(
        semseg_trained_model,
        use_folds=semseg_folds,
        checkpoint_name='checkpoint_final.pth'
    )
    semseg_pred = pred_semseg.predict_single_npy_array(im, prop, None, use_memory_mapping)
    
    # 명시적 메모리 정리
    del pred_semseg
    torch.cuda.empty_cache()
    gc.collect()
    
    return semseg_pred


def map_labels_to_toothfairy(predicted_seg: np.ndarray) -> np.ndarray:
    """
    nnU-Net 예측 결과 라벨을 대회 규정에 맞는 라벨로 변경하는 함수.
    """
    max_label = 77
    mapping = np.arange(max_label + 1, dtype=np.int32)
    remapping_dict = {
        19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 26, 25: 27, 26: 28,
        27: 31, 28: 32, 29: 33, 30: 34, 31: 35, 32: 36, 33: 37, 34: 38,
        35: 41, 36: 42, 37: 43, 38: 44, 39: 45, 40: 46, 41: 47, 42: 48,
        43: 51, 44: 52, 45: 53,
    }
    for old_label, new_label in remapping_dict.items():
        mapping[old_label] = new_label
    mapping[46:78] = 50
    return mapping[predicted_seg]


def postprocess(prediction_npy, vol_per_voxel, verbose: bool = False):
    """
    예측 결과에서 특정 크기 미만의 작은 노이즈 객체들을 제거하는 후처리 함수.
    """
    cutoffs = {1: 0.0, 2: 78411.5, 3: 0.0, 4: 0.0, 5: 2800.0, 6: 1216.5, 7: 0.0, 8: 6222.0, 
               9: 1573.0, 10: 946.0, 11: 0.0, 12: 6783.5, 13: 9469.5, 14: 0.0, 15: 2260.0, 
               16: 3566.0, 17: 6321.0, 18: 4221.5, 19: 5829.0, 20: 0.0, 21: 0.0, 22: 468.0, 
               23: 1555.0, 24: 1291.5, 25: 2834.5, 26: 584.5, 27: 0.0, 28: 0.0, 29: 0.0, 
               30: 0.0, 31: 1935.5, 32: 0.0, 33: 0.0, 34: 6140.0, 35: 0.0, 36: 0.0, 37: 0.0, 
               38: 2710.0, 39: 0.0, 40: 0.0, 41: 0.0, 42: 970.0}
    vol_per_voxel_cutoffs = 0.3 * 0.3 * 0.3
    for c, co in cutoffs.items():
        if co > 0:
            mask = prediction_npy == c
            pred_vol = np.sum(mask) * vol_per_voxel
            if 0 < pred_vol < (co * vol_per_voxel_cutoffs):
                prediction_npy[mask] = 0
                if verbose:
                    print(f'removed label {c}')
    return prediction_npy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=Path, default="/input/images/cbct/")
    parser.add_argument('-o', '--output_folder', type=Path, default="/output/images/oral-pharyngeal-segmentation/")
    parser.add_argument('-sem_mod', '--semseg_trained_model', type=str, default="/opt/algorithm/nnunet_model")
    parser.add_argument('--semseg_folds', type=str, nargs='+', default=[0, 1])
    parser.add_argument('--use_memory_mapping', action='store_true', 
                        help='Use disk-based memory mapping for extreme memory saving (slower)')
    args = parser.parse_args()

    args.output_folder.mkdir(exist_ok=True, parents=True)
    semseg_folds = [i if i == 'all' else int(i) for i in args.semseg_folds]
    rw = SimpleITKIO()
    input_files = list(args.input_folder.glob('*.nii.gz')) + list(args.input_folder.glob('*.mha'))

    for input_fname in input_files:
        print(f"\nProcessing: {input_fname.name}")
        output_fname = args.output_folder / input_fname.name

        # 이미지 로드
        im, prop = rw.read_images([input_fname])

        # 예측 실행
        with torch.no_grad():
            semseg_pred = predict_semseg(im, prop, args.semseg_trained_model, 
                                       semseg_folds, args.use_memory_mapping)

        # 후처리 및 라벨 변환
        # semseg_pred = postprocess(semseg_pred, np.prod(prop['spacing']), True)
        semseg_pred = map_labels_to_toothfairy(semseg_pred)
        
        # 결과 저장
        rw.write_seg(semseg_pred, output_fname, prop)

        # 메모리 정리
        del im, prop, semseg_pred
        gc.collect()
        
        print(f"Finished: {input_fname.name}")