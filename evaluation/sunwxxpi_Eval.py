import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nb
import torch
from collections import OrderedDict
from tqdm import tqdm
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric

join = os.path.join
basename = os.path.basename

parser = argparse.ArgumentParser()
parser.add_argument(
    '--gt_path',
    type=str,
    required=True,
    help='경고: 실제 레이블 파일이 있는 경로 (예: ./data/nnUNet_raw/Dataset001_COCA/labelsVal)'
)
parser.add_argument(
    '--seg_path',
    type=str,
    required=True,
    help='경고: 예측된 세그멘테이션 파일이 있는 경로 (예: ./results/001_COCA/3d_0.0005)'
)
parser.add_argument(
    '--save_path',
    type=str,
    required=True,
    help='결과를 저장할 CSV 파일 경로 (예: 3d_0.0005_DSC_AP_HD.csv)'
)

args = parser.parse_args()

gt_path = args.gt_path
seg_path = args.seg_path
save_path = args.save_path

filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.nii.gz')]
filenames = [x for x in filenames if os.path.exists(join(seg_path, x))]
filenames.sort()

dice_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=True)
miou_metric = MeanIoU(include_background=True, reduction="mean", ignore_empty=True)
hd_metric = HausdorffDistanceMetric(include_background=True, distance_metric="euclidean", percentile=95)

seg_metrics = OrderedDict()
seg_metrics['Name'] = list()
""" label_tolerance = OrderedDict({'Left Coronary Artery': 1, 
                                'Left Anterior Descending Artery': 2, 
                                'Left Circumflex Artery': 3, 
                                'Right Coronary Artery': 4}) """
label_tolerance = OrderedDict({'Aortic Valve Calcium': 1})

for cardiovascular in label_tolerance.keys():
    seg_metrics[f'{cardiovascular}_Dice'] = list()
    seg_metrics[f'{cardiovascular}_IoU'] = list()
    seg_metrics[f'{cardiovascular}_HD95'] = list()

seg_metrics['Average_Dice'] = list()
seg_metrics['Average_IoU'] = list()
seg_metrics['Average_HD95'] = list()

classwise_dice = {key: [] for key in label_tolerance.keys()}
classwise_iou = {key: [] for key in label_tolerance.keys()}
classwise_hd = {key: [] for key in label_tolerance.keys()}

for name in tqdm(filenames, desc="Processing files"):
    seg_metrics['Name'].append(name)
    
    gt_nii = nb.load(join(gt_path, name))
    seg_nii = nb.load(join(seg_path, name))
    gt_data = np.uint8(gt_nii.get_fdata())
    seg_data = np.uint8(seg_nii.get_fdata())

    gt_tensor = torch.from_numpy(gt_data).unsqueeze(0).unsqueeze(0)
    seg_tensor = torch.from_numpy(seg_data).unsqueeze(0).unsqueeze(0)

    average_dice = []
    average_iou = []
    average_hd = []

    for cardiovascular, label in label_tolerance.items():
        gt_binary = (gt_tensor == label).float()
        seg_binary = (seg_tensor == label).float()

        dice = dice_metric(y_pred=seg_binary, y=gt_binary)
        iou = miou_metric(y_pred=seg_binary, y=gt_binary)
        hd = hd_metric(y_pred=seg_binary, y=gt_binary)
        
        if isinstance(hd, torch.Tensor):
            hd = hd.item()

        seg_metrics[f'{cardiovascular}_Dice'].append(round(dice.item(), 4))
        seg_metrics[f'{cardiovascular}_IoU'].append(round(iou.item(), 4))
        seg_metrics[f'{cardiovascular}_HD95'].append(round(hd, 2))

        classwise_dice[cardiovascular].append(dice.item())
        classwise_iou[cardiovascular].append(iou.item())
        classwise_hd[cardiovascular].append(hd)

        average_dice.append(dice.item())
        average_iou.append(iou.item())
        average_hd.append(hd)

    avg_dice = np.nanmean(average_dice)
    avg_iou = np.nanmean(average_iou)
    avg_hd = np.nanmean(average_hd)
    
    seg_metrics['Average_Dice'].append(round(avg_dice, 4))
    seg_metrics['Average_IoU'].append(round(avg_iou, 4))
    seg_metrics['Average_HD95'].append(round(avg_hd, 2))

dataframe = pd.DataFrame(seg_metrics)
dataframe.to_csv(save_path, index=False)
print("Complete!!")

print("\nMean Metrics per Class:")
for cardiovascular in label_tolerance.keys():
    avg_dice = np.nanmean(classwise_dice[cardiovascular])
    avg_iou = np.nanmean(classwise_iou[cardiovascular])
    avg_hd = np.nanmean(classwise_hd[cardiovascular])

    print(f'{cardiovascular}: Dice = {avg_dice:.4f}, IoU = {avg_iou:.4f}, HD95 = {avg_hd:.2f}')

overall_avg_dice = np.nanmean([np.nanmean(v) for v in classwise_dice.values()])
overall_avg_iou = np.nanmean([np.nanmean(v) for v in classwise_iou.values()])
overall_avg_hd = np.nanmean([np.nanmean(v) for v in classwise_hd.values()])

print("\n" + 20 * '>' )
print(f'Overall Average Dice : {overall_avg_dice:.4f}')
print(f'Overall Average IoU : {overall_avg_iou:.4f}')
print(f'Overall Average HD95 : {overall_avg_hd:.2f}')
print(20 * '<')