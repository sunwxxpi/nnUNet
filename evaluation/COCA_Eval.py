import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nb
from collections import OrderedDict
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
from scipy.spatial.distance import directed_hausdorff

# Dice coefficient 계산 함수
def compute_dice_coefficient(mask_gt, mask_pred):
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2 * volume_intersect / volume_sum

# Average Precision (AP) 계산 함수
def compute_average_precision(mask_gt, mask_pred):
    precision, recall, _ = precision_recall_curve(mask_gt.flatten(), mask_pred.flatten())
    return auc(recall, precision)

# Hausdorff Distance 계산 함수
def compute_hausdorff_distance(mask_gt, mask_pred):
    gt_points = np.transpose(np.nonzero(mask_gt))
    pred_points = np.transpose(np.nonzero(mask_pred))
    if len(gt_points) == 0 or len(pred_points) == 0:
        return np.NaN
    hd_1 = directed_hausdorff(gt_points, pred_points)[0]
    hd_2 = directed_hausdorff(pred_points, gt_points)[0]
    return max(hd_1, hd_2)

# 각 사례(case)별로 메트릭 계산
def calculate_metric_percase(gt_mask, seg_mask):
    seg_mask[seg_mask > 0] = 1
    gt_mask[gt_mask > 0] = 1

    # GT와 Segmentation 모두 병변이 없을 경우
    if gt_mask.sum() == 0 and seg_mask.sum() == 0:
        dice = 1
        m_ap = 1
        hd = 0
    # GT에는 병변이 없는데 Segmentation에는 병변이 있을 경우
    elif gt_mask.sum() == 0 and seg_mask.sum() > 0:
        dice = 0
        m_ap = 0
        hd = np.NaN
    # 일반적인 경우
    else:
        dice = compute_dice_coefficient(gt_mask, seg_mask)
        m_ap = compute_average_precision(gt_mask, seg_mask)
        hd = compute_hausdorff_distance(gt_mask, seg_mask)
        
    return dice, m_ap, hd

# 파일 경로 관련 함수 정의
join = os.path.join
basename = os.path.basename

# 명령행 인자 파싱
parser = argparse.ArgumentParser()
parser.add_argument('--gt_path',   type=str, default='')
parser.add_argument('--seg_path',  type=str, default='')
parser.add_argument('--save_path', type=str, default='')
args = parser.parse_args()

# 경로 설정
gt_path = args.gt_path
seg_path = args.seg_path
save_path = args.save_path

# Segmentation 결과 파일 리스트 불러오기
filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.nii.gz')]
filenames = [x for x in filenames if os.path.exists(join(seg_path, x))]
filenames.sort()

# Label 정의
label_tolerance = OrderedDict({
    'Left Coronary Artery': 1, 
    'Left Anterior Descending Artery': 2, 
    'Left Circumflex Artery': 3, 
    'Right Coronary Artery': 4
})

# 3D 평가 결과 저장용 딕셔너리
seg_metrics = OrderedDict()
seg_metrics['Name'] = list()

for cardiovascular in label_tolerance.keys():
    seg_metrics[f'{cardiovascular}_DSC'] = list()
    seg_metrics[f'{cardiovascular}_AP']  = list()
    seg_metrics[f'{cardiovascular}_HD']  = list()

seg_metrics['Average_DSC'] = list()
seg_metrics['Average_AP']  = list()
seg_metrics['Average_HD']  = list()

# 2D 슬라이스 레벨 평가 결과 저장용 딕셔너리
seg_metrics_2D = OrderedDict()
for cardiovascular in label_tolerance.keys():
    seg_metrics_2D[f'{cardiovascular}_2D_sliceCount'] = list()
    seg_metrics_2D[f'{cardiovascular}_2D_DSC'] = list()
    seg_metrics_2D[f'{cardiovascular}_2D_AP']  = list()

seg_metrics_2D['Average_2D_DSC'] = list()
seg_metrics_2D['Average_2D_AP']  = list()

for name in tqdm(filenames):
    # (A) 3D 볼륨 데이터 로드
    gt_nii = nb.load(join(gt_path, name))
    seg_nii = nb.load(join(seg_path, name))
    gt_data = np.uint8(gt_nii.get_fdata())   # (D, H, W) 형태로 로드
    seg_data = np.uint8(seg_nii.get_fdata())

    seg_metrics['Name'].append(name)

    # (A-1) 3D 메트릭 계산
    average_DSC_3d = []
    average_AP_3d  = []
    average_HD_3d  = []

    for cardiovascular, label_idx in label_tolerance.items():
        organ_i_gt  = (gt_data == label_idx)
        organ_i_seg = (seg_data == label_idx)

        # 3D 메트릭 계산
        dice_3d, ap_3d, hd_3d = calculate_metric_percase(organ_i_gt, organ_i_seg)

        # 메트릭 결과 저장
        seg_metrics[f'{cardiovascular}_DSC'].append(round(dice_3d, 4))
        seg_metrics[f'{cardiovascular}_AP'].append(round(ap_3d, 4))
        seg_metrics[f'{cardiovascular}_HD'].append(round(hd_3d, 4))

        # 평균 계산용 리스트에 추가
        average_DSC_3d.append(dice_3d)
        average_AP_3d.append(ap_3d)
        average_HD_3d.append(hd_3d)

    # 3D 메트릭 평균 계산
    avg_dsc_3d = np.nanmean(average_DSC_3d)
    avg_ap_3d  = np.nanmean(average_AP_3d)
    avg_hd_3d  = np.nanmean(average_HD_3d)

    seg_metrics['Average_DSC'].append(round(avg_dsc_3d, 4))
    seg_metrics['Average_AP'].append(round(avg_ap_3d, 4))
    seg_metrics['Average_HD'].append(round(avg_hd_3d, 4))

    # (A-2) 2D 슬라이스 레벨 메트릭 계산
    slice_DSC_dict = {k: [] for k in label_tolerance.keys()}
    slice_AP_dict  = {k: [] for k in label_tolerance.keys()}
    slice_count_dict = {k: 0 for k in label_tolerance.keys()}

    D = gt_data.shape[0]  # 깊이(Dimension) 정보
    for d in range(D):
        slice_gt  = gt_data[d, :, :]
        slice_seg = seg_data[d, :, :]

        # 각 label에 대해 슬라이스 메트릭 계산
        for cardiovascular, label_idx in label_tolerance.items():
            gt_mask_2d  = (slice_gt  == label_idx)
            seg_mask_2d = (slice_seg == label_idx)

            # 병변이 있는 경우에만 계산
            if gt_mask_2d.sum() > 0:
                slice_count_dict[cardiovascular] += 1
                dice_2d, ap_2d, _ = calculate_metric_percase(gt_mask_2d, seg_mask_2d)
                slice_DSC_dict[cardiovascular].append(dice_2d)
                slice_AP_dict[cardiovascular].append(ap_2d)

    # 각 label 별 2D 메트릭 평균 계산
    sum_dsc_2d = []
    sum_ap_2d  = []
    for cardiovascular in label_tolerance.keys():
        if len(slice_DSC_dict[cardiovascular]) == 0:
            seg_metrics_2D[f'{cardiovascular}_2D_sliceCount'].append(0)
            seg_metrics_2D[f'{cardiovascular}_2D_DSC'].append(np.nan)
            seg_metrics_2D[f'{cardiovascular}_2D_AP'].append(np.nan)
        else:
            d_mean = np.nanmean(slice_DSC_dict[cardiovascular])
            ap_mean= np.nanmean(slice_AP_dict[cardiovascular])

            seg_metrics_2D[f'{cardiovascular}_2D_sliceCount'].append(slice_count_dict[cardiovascular])
            seg_metrics_2D[f'{cardiovascular}_2D_DSC'].append(round(d_mean, 4))
            seg_metrics_2D[f'{cardiovascular}_2D_AP'].append(round(ap_mean, 4))

            sum_dsc_2d.append(d_mean)
            sum_ap_2d.append(ap_mean)

    if len(sum_dsc_2d) > 0:
        avg_dsc_2d = np.nanmean(sum_dsc_2d)
        avg_ap_2d  = np.nanmean(sum_ap_2d)
    else:
        avg_dsc_2d = np.nan
        avg_ap_2d  = np.nan

    seg_metrics_2D['Average_2D_DSC'].append(round(avg_dsc_2d, 4))
    seg_metrics_2D['Average_2D_AP'].append(round(avg_ap_2d, 4))

# (3) 결과 데이터프레임으로 저장
dataframe_3D = pd.DataFrame(seg_metrics)
dataframe_2D = pd.DataFrame(seg_metrics_2D)

result_merged = pd.concat([dataframe_3D, dataframe_2D], axis=1)
result_merged.to_csv(save_path, index=False)
print("Complete!!")

# (4) 3D 메트릭 평균 출력
case_avg_DSC = dataframe_3D.filter(regex='_DSC$').mean(axis=0)
case_avg_AP  = dataframe_3D.filter(regex='_AP$').mean(axis=0)
case_avg_HD  = dataframe_3D.filter(regex='_HD$').mean(axis=0)

print(20 * '>')
print(f"Average DSC for {basename(seg_path)}: {case_avg_DSC.mean():.4f}")
print(f"Average AP  for {basename(seg_path)}: {case_avg_AP.mean():.4f}")
print(f"Average HD  for {basename(seg_path)}: {case_avg_HD.mean():.4f}")

print("\nCardiovascular-wise Average (3D) DSC, AP, and HD:")
for cardiovascular in label_tolerance.keys():
    cardiovascular_avg_dsc = dataframe_3D[f'{cardiovascular}_DSC'].mean()
    cardiovascular_avg_ap  = dataframe_3D[f'{cardiovascular}_AP'].mean()
    cardiovascular_avg_hd  = dataframe_3D[f'{cardiovascular}_HD'].mean()
    print(f"{cardiovascular}: DSC={cardiovascular_avg_dsc:.4f}, AP={cardiovascular_avg_ap:.4f}, HD={cardiovascular_avg_hd:.4f}")
print(20 * '<')

# (5) 2D 메트릭 평균 출력 (병변 존재 슬라이스만)
dsc_cols_2d = [col for col in dataframe_2D.columns if col.endswith('_2D_DSC')]
ap_cols_2d  = [col for col in dataframe_2D.columns if col.endswith('_2D_AP')]

case_avg_DSC_2d = dataframe_2D[dsc_cols_2d].mean(axis=0)
case_avg_AP_2d  = dataframe_2D[ap_cols_2d].mean(axis=0)

print(20 * '>')
print(f"Average DSC for {basename(seg_path)} [LesionOnly]: {case_avg_DSC_2d.mean():.4f}")
print(f"Average AP  for {basename(seg_path)} [LesionOnly]: {case_avg_AP_2d.mean():.4f}")

print("\nCardiovascular-wise Average (2D [LesionOnly]) DSC, AP:")
for cardiovascular in label_tolerance.keys():
    dsc_col_name = f"{cardiovascular}_2D_DSC"
    ap_col_name  = f"{cardiovascular}_2D_AP"
    if dsc_col_name in dataframe_2D.columns and ap_col_name in dataframe_2D.columns:
        dsc_val_2d = dataframe_2D[dsc_col_name].mean()
        ap_val_2d  = dataframe_2D[ap_col_name].mean()
        print(f"{cardiovascular}: DSC={dsc_val_2d:.4f}, AP={ap_val_2d:.4f}")
    else:
        print(f"{cardiovascular}: No 2D data found (no lesion slices).")
print(20 * '<')