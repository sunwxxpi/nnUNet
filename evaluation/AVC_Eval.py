import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nb
import torch
from collections import OrderedDict
from tqdm import tqdm
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric

# 자주 사용하는 os.path 함수 정의
join = os.path.join
basename = os.path.basename

# --- Argument Parser 설정 ---
parser = argparse.ArgumentParser(description="Evaluate 3D medical image segmentation metrics.")
parser.add_argument(
    '--gt_path',
    type=str,
    required=True,
    help='Path to the directory containing ground truth label files (e.g., ./data/labelsVal)'
)
parser.add_argument(
    '--seg_path',
    type=str,
    required=True,
    help='Path to the directory containing predicted segmentation files (e.g., ./results/pred)'
)
parser.add_argument(
    '--save_path',
    type=str,
    required=True,
    help='File path to save the results CSV (e.g., ./results/evaluation.csv)'
)

args = parser.parse_args()

# --- 경로 및 파일 목록 준비 ---
gt_path = args.gt_path
seg_path = args.seg_path
save_path = args.save_path

filenames = os.listdir(seg_path)
filenames = [f for f in filenames if f.endswith(('.nii.gz', '.nii'))]
# 정답 파일이 존재하는 경우만 평가 목록에 포함
filenames = [f for f in filenames if os.path.exists(join(gt_path, f))]
filenames.sort()

# --- MONAI 메트릭 초기화 ---
# include_background=True는 여기서 큰 의미가 없음 (어차피 클래스별로 이진화하여 계산)
# ignore_empty=True: GT와 Pred가 모두 비어있는 경우(True Positive가 0) 계산에서 제외
dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=True)
miou_metric = MeanIoU(include_background=True, reduction="mean", ignore_empty=True)
hd_metric = HausdorffDistanceMetric(include_background=True, distance_metric="euclidean", percentile=95)

# --- 결과 저장을 위한 OrderedDict 초기화 ---
seg_metrics = OrderedDict()
seg_metrics['Name'] = list()

label_tolerance = OrderedDict({
    "aortic valve calcium": 1
})

# DataFrame의 컬럼(열) 순서를 정의
for cardiovascular in label_tolerance.keys():
    seg_metrics[f'{cardiovascular}_Dice'] = list()
    seg_metrics[f'{cardiovascular}_IoU'] = list()
    seg_metrics[f'{cardiovascular}_HD95'] = list()

seg_metrics['Average_Dice'] = list()
seg_metrics['Average_IoU'] = list()
seg_metrics['Average_HD95'] = list()

# --- 각 파일에 대한 메트릭 계산 루프 ---
for name in tqdm(filenames, desc="Evaluating segmentation files"):
    seg_metrics['Name'].append(name)
    
    # NIfTI 파일 로드
    gt_nii = nb.load(join(gt_path, name))
    seg_nii = nb.load(join(seg_path, name))
    
    # 데이터 타입 안정성 강화: uint8 대신 int32 사용
    gt_data = gt_nii.get_fdata().astype(np.int32)
    seg_data = seg_nii.get_fdata().astype(np.int32)

    # PyTorch 텐서로 변환 (MONAI는 B, C, H, W, D 형태를 기대)
    gt_tensor = torch.from_numpy(gt_data).int().unsqueeze(0).unsqueeze(0)
    seg_tensor = torch.from_numpy(seg_data).int().unsqueeze(0).unsqueeze(0)

    # 현재 파일의 클래스별 메트릭을 저장할 임시 리스트
    per_file_dice = []
    per_file_iou = []
    per_file_hd = []

    for cardiovascular, label in label_tolerance.items():
        # 각 클래스에 대해 이진(binary) 마스크 생성
        gt_binary = (gt_tensor == label).float()
        seg_binary = (seg_tensor == label).float()

        # 메트릭 계산
        dice = dice_metric(y_pred=seg_binary, y=gt_binary).item()
        iou = miou_metric(y_pred=seg_binary, y=gt_binary).item()
        hd = hd_metric(y_pred=seg_binary, y=gt_binary).item()
        
        # 계산된 메트릭을 딕셔너리에 추가
        seg_metrics[f'{cardiovascular}_Dice'].append(round(dice, 4))
        seg_metrics[f'{cardiovascular}_IoU'].append(round(iou, 4))
        seg_metrics[f'{cardiovascular}_HD95'].append(round(hd, 2))

        # 현재 파일의 평균 계산을 위해 저장 (NaN이 아닌 경우만)
        if not np.isnan(dice): per_file_dice.append(dice)
        if not np.isnan(iou): per_file_iou.append(iou)
        if not np.isnan(hd): per_file_hd.append(hd)

    # 현재 파일의 평균 메트릭 계산 및 저장
    avg_dice = np.mean(per_file_dice) if per_file_dice else 0.0
    avg_iou = np.mean(per_file_iou) if per_file_iou else 0.0
    avg_hd = np.mean(per_file_hd) if per_file_hd else 0.0
    
    seg_metrics['Average_Dice'].append(round(avg_dice, 4))
    seg_metrics['Average_IoU'].append(round(avg_iou, 4))
    seg_metrics['Average_HD95'].append(round(avg_hd, 2))

# --- 결과 저장 및 통계 출력 ---
dataframe = pd.DataFrame(seg_metrics)
dataframe.to_csv(save_path, index=False)
print(f"\nEvaluation results have been saved to {save_path}")

print("\n" + "="*25)
print("  Mean Metrics per Class")
print("="*25)
for cardiovascular in label_tolerance.keys():
    # DataFrame에서 직접 클래스별 평균 계산
    mean_dice = dataframe[f'{cardiovascular}_Dice'].mean()
    mean_iou = dataframe[f'{cardiovascular}_IoU'].mean()
    mean_hd = dataframe[f'{cardiovascular}_HD95'].mean()
    print(f'{cardiovascular:<33}: Dice={mean_dice:.4f}, IoU={mean_iou:.4f}, HD95={mean_hd:.2f}')

# 전체 평균 계산
overall_avg_dice = dataframe['Average_Dice'].mean()
overall_avg_iou = dataframe['Average_IoU'].mean()
overall_avg_hd = dataframe['Average_HD95'].mean()

print("\n" + 20 * '>')
print(f' Overall Average Dice : {overall_avg_dice:.4f}')
print(f' Overall Average IoU  : {overall_avg_iou:.4f}')
print(f' Overall Average HD95 : {overall_avg_hd:.2f}')
print(20 * '<')