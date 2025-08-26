import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import SimpleITK as sitk
import torch
from evalutils import SegmentationAlgorithm
from evalutils.validators import UniqueImagesValidator, UniquePathIndicesValidator
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# --- Configuration Constants ---
# 입력 및 출력 경로 설정
INPUT_DIR = Path("/input/images/cbct/")
OUTPUT_SEG_DIR = Path("/output/images/oral-pharyngeal-segmentation/")
OUTPUT_META_DIR = Path("/output/metadata/")

# 모델 경로 설정
DOCKER_MODEL_PATH = "/opt/algorithm/nnunet_model"
DEV_MODEL_PATH = "/home/psw/nnUNet/data/nnUNet_results/ToothFairy_Final/nnUNetTrainer__nnUNetResEncUNetLPlans_torchres__3d_fullres"
CHECKPOINT_NAME = "checkpoint_final.pth"


def get_default_device() -> torch.device:
    """사용 가능한 경우 CUDA 장치를, 그렇지 않은 경우 CPU를 반환합니다."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def map_labels_to_toothfairy(predicted_seg: np.ndarray) -> np.ndarray:
    """
    NumPy 룩업 테이블을 사용하여 nnU-Net 라벨을 ToothFairy FDI 라벨로 효율적으로 매핑합니다.

    Args:
        predicted_seg: nnU-Net 모델의 원본 세그멘테이션 예측 결과입니다.

    Returns:
        FDI 라벨링 시스템으로 리매핑된 세그멘테이션 마스크입니다.
    """
    # nnU-Net 라벨(인덱스)을 ToothFairy FDI 라벨(값)으로 매핑
    label_mapping = {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9,
        10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18,
        19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 26, 25: 27, 26: 28,
        27: 31, 28: 32, 29: 33, 30: 34, 31: 35, 32: 36, 33: 37, 34: 38,
        35: 41, 36: 42, 37: 43, 38: 44, 39: 45, 40: 46, 41: 47, 42: 48,
        43: 51, 44: 52, 45: 53,
    }
    # 46부터 77까지의 모든 Pulp 라벨을 단일 클래스 50으로 매핑
    for pulp_label in range(46, 78):
        label_mapping[pulp_label] = 50

    # NumPy 배열을 룩업 테이블로 사용하여 빠른 변환 수행
    max_label = max(label_mapping.keys())
    lookup_table = np.zeros(max_label + 1, dtype=np.int16)
    for nnunet_label, toothfairy_label in label_mapping.items():
        lookup_table[nnunet_label] = toothfairy_label

    return lookup_table[predicted_seg]


class ToothFairy3_OralPharyngealSegmentation(SegmentationAlgorithm):
    """CBCT 이미지에서 구강 및 인두 영역을 분할하는 알고리즘입니다."""

    def __init__(self):
        super().__init__(
            input_path=INPUT_DIR,
            output_path=OUTPUT_SEG_DIR,
            validators=dict(
                input_image=(UniqueImagesValidator(), UniquePathIndicesValidator())
            ),
        )

        # 출력 디렉터리 생성
        self._output_path.mkdir(parents=True, exist_ok=True)
        self.metadata_output_path = self._setup_metadata_dir()

        # 장치 및 nnU-Net 환경 변수 설정
        self.device = get_default_device()
        self._setup_nnunet_environment()
        print(f"Using device: {self.device}")

        # nnU-Net 예측기 초기화
        self.predictor = self._initialize_predictor()

    def _setup_metadata_dir(self) -> Path:
        """메타데이터 출력 디렉터리를 설정하고 권한 오류 시 대체 경로를 사용합니다."""
        try:
            OUTPUT_META_DIR.mkdir(parents=True, exist_ok=True, mode=0o777)
            return OUTPUT_META_DIR
        except (PermissionError, OSError) as e:
            print(f"Warning: Cannot create {OUTPUT_META_DIR}: {e}. Using /tmp.")
            fallback_dir = Path("/tmp/metadata/")
            fallback_dir.mkdir(parents=True, exist_ok=True)
            return fallback_dir

    def _setup_nnunet_environment(self):
        """Docker 환경에 필요한 nnU-Net 환경 변수를 설정합니다."""
        env_vars = {
            "nnUNet_raw": "/tmp/nnUNet_raw",
            "nnUNet_preprocessed": "/tmp/nnUNet_preprocessed",
            "nnUNet_results": "/opt/algorithm",
        }
        for key, value in env_vars.items():
            if key not in os.environ:
                os.environ[key] = value
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)

    def _initialize_predictor(self) -> nnUNetPredictor:
        """모델 경로를 탐색하고 nnU-Net 예측기를 초기화합니다."""
        possible_paths: List[str] = [DOCKER_MODEL_PATH, DEV_MODEL_PATH]
        model_folder = None

        for path_str in possible_paths:
            path = Path(path_str)
            if (path / "fold_0" / CHECKPOINT_NAME).exists():
                model_folder = str(path)
                print(f"nnU-Net model found at: {model_folder}")
                break

        if model_folder is None:
            raise FileNotFoundError(f"Could not find model in {possible_paths}")

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        predictor.initialize_from_trained_model_folder(
            model_folder, use_folds=("all",), checkpoint_name=CHECKPOINT_NAME
        )
        return predictor

    def save_instance_metadata(self, metadata: Dict, image_name: str):
        """인스턴스 메타데이터를 JSON 파일로 저장합니다."""
        metadata_file = self.metadata_output_path / f"{image_name}_instances.json"
        try:
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to: {metadata_file}")
        except (PermissionError, OSError) as e:
            print(f"Warning: Failed to save metadata to {metadata_file}: {e}")

    @torch.no_grad()
    def predict(self, *, input_image: sitk.Image) -> sitk.Image:
        """단일 입력 이미지에 대해 세그멘테이션을 수행합니다."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            # 입력 이미지를 임시 파일로 저장
            input_filepath = input_dir / "image_0000.nii.gz"
            sitk.WriteImage(input_image, str(input_filepath))

            # nnU-Net 추론 실행
            print("Running nnU-Net segmentation...")
            self.predictor.predict_from_files(
                list_of_lists_or_source_folder=str(input_dir),
                output_folder_or_list_of_truncated_output_files=str(output_dir),
                save_probabilities=False,
                num_processes_preprocessing=8,
                num_processes_segmentation_export=8,
            )

            # 결과 읽기 및 후처리
            output_filepath = output_dir / "image.nii.gz"
            if not output_filepath.exists():
                raise FileNotFoundError(f"Segmentation result not found at {output_filepath}")

            output_sitk = sitk.ReadImage(str(output_filepath))
            output_array = sitk.GetArrayFromImage(output_sitk)

        # 라벨 매핑 및 SimpleITK 이미지 생성
        remapped_array = map_labels_to_toothfairy(output_array)
        output_image = sitk.GetImageFromArray(remapped_array)
        output_image.CopyInformation(input_image)

        # 최종 이미지의 픽셀 타입을 정수형(UInt8)으로 명시적으로 변환하여 안정성 확보
        output_image = sitk.Cast(output_image, sitk.sitkUInt8)

        # 메타데이터 생성 및 저장
        unique_labels = np.unique(remapped_array)
        instances = [{"label": int(label)} for label in unique_labels if label > 0]
        instance_metadata = {"instances": instances, "total_instances": len(instances)}
        self.save_instance_metadata(instance_metadata, "prediction")

        print("Segmentation completed successfully.")
        return output_image


if __name__ == "__main__":
    ToothFairy3_OralPharyngealSegmentation().process()