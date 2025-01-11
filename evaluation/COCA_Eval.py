import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nb
from collections import OrderedDict
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
from scipy.spatial.distance import directed_hausdorff

def compute_dice_coefficient(mask_gt, mask_pred):
    """Compute Soerensen-Dice coefficient."""
    volume_sum = mask_gt.sum() + mask_pred.sum()
    
    if volume_sum == 0:
        return np.NaN
    
    volume_intersect = (mask_gt & mask_pred).sum()
    
    return 2 * volume_intersect / volume_sum

def compute_average_precision(mask_gt, mask_pred):
    """Compute Average Precision (AP) score."""
    precision, recall, _ = precision_recall_curve(mask_gt.flatten(), mask_pred.flatten())
    
    return auc(recall, precision)

def compute_hausdorff_distance(mask_gt, mask_pred):
    """Compute Hausdorff Distance (HD)."""
    gt_points = np.transpose(np.nonzero(mask_gt))
    pred_points = np.transpose(np.nonzero(mask_pred))
    
    if len(gt_points) == 0 or len(pred_points) == 0:
        return np.NaN
    
    hd_1 = directed_hausdorff(gt_points, pred_points)[0]
    hd_2 = directed_hausdorff(pred_points, gt_points)[0]
    
    return max(hd_1, hd_2)

join = os.path.join
basename = os.path.basename

parser = argparse.ArgumentParser()
parser.add_argument(
    '--gt_path',
    type=str,
    default=''
)
parser.add_argument(
    '--seg_path',
    type=str,
    default=''
)
parser.add_argument(
    '--save_path',
    type=str,
    default=''
)

args = parser.parse_args()

gt_path = args.gt_path  # ./data/nnUNet_raw/Dataset001_COCA/labelsVal
seg_path = args.seg_path  # ./results/001_COCA/3d_0.0005
save_path = args.save_path  # 3d_0.0005_DSC_AP_HD.csv

filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.nii.gz')]
filenames = [x for x in filenames if os.path.exists(join(seg_path, x))]
filenames.sort()

seg_metrics = OrderedDict()
seg_metrics['Name'] = list()
label_tolerance = OrderedDict({'Left Coronary Artery': 1, 
                               'Left Anterior Descending Artery': 2, 
                               'Left Circumflex Artery': 3, 
                               'Right Coronary Artery': 4})

for cardiovascular in label_tolerance.keys():
    seg_metrics[f'{cardiovascular}_DSC'] = list()
    seg_metrics[f'{cardiovascular}_AP'] = list()
    seg_metrics[f'{cardiovascular}_HD'] = list()

seg_metrics['Average_DSC'] = list()
seg_metrics['Average_AP'] = list()
seg_metrics['Average_HD'] = list()

for name in tqdm(filenames):
    seg_metrics['Name'].append(name)
    
    gt_nii = nb.load(join(gt_path, name))
    seg_nii = nb.load(join(seg_path, name))
    gt_data = np.uint8(gt_nii.get_fdata())
    seg_data = np.uint8(seg_nii.get_fdata())

    average_DSC = []
    average_AP = []
    average_HD = []

    for i, cardiovascular in enumerate(label_tolerance.keys(), start=1):
        if np.sum(gt_data == i) == 0 and np.sum(seg_data == i) == 0:
            DSC_i = 1
            AP_i = 1
            HD_i = 0
        elif np.sum(gt_data == i) == 0 and np.sum(seg_data == i) > 0:
            DSC_i = 0
            AP_i = 0
            HD_i = np.NaN
        else:
            organ_i_gt = (gt_data == i)
            organ_i_seg = (seg_data == i)
            
            DSC_i = compute_dice_coefficient(organ_i_gt, organ_i_seg)
            AP_i = compute_average_precision(organ_i_gt, organ_i_seg)
            HD_i = compute_hausdorff_distance(organ_i_gt, organ_i_seg)
        
        average_DSC.append(DSC_i)
        average_AP.append(AP_i)
        average_HD.append(HD_i)

        seg_metrics['{}_DSC'.format(cardiovascular)].append(round(DSC_i, 4))
        seg_metrics['{}_AP'.format(cardiovascular)].append(round(AP_i, 4))
        seg_metrics['{}_HD'.format(cardiovascular)].append(round(HD_i, 4))

    avg_dsc = np.nanmean(average_DSC)
    avg_ap = np.nanmean(average_AP)
    avg_hd = np.nanmean(average_HD)
    
    seg_metrics['Average_DSC'].append(round(avg_dsc, 4))
    seg_metrics['Average_AP'].append(round(avg_ap, 4))
    seg_metrics['Average_HD'].append(round(avg_hd, 4))

dataframe = pd.DataFrame(seg_metrics)
dataframe.to_csv(save_path, index=False)
print("Complete!!")

case_avg_DSC = dataframe.filter(regex='_DSC$').mean(axis=0)
case_avg_AP = dataframe.filter(regex='_AP$').mean(axis=0)
case_avg_HD = dataframe.filter(regex='_HD$').mean(axis=0)

print(20 * '>')
print(f'Average DSC for {basename(seg_path)}: {case_avg_DSC.mean()}')
print(f'Average AP for {basename(seg_path)}: {case_avg_AP.mean()}')
print(f'Average HD for {basename(seg_path)}: {case_avg_HD.mean()}')
print("\nCardiovascular-wise Average DSC, AP, and HD: ")
for cardiovascular in label_tolerance.keys():
    cardiovascular_avg_dsc = dataframe['{}_DSC'.format(cardiovascular)].mean()
    cardiovascular_avg_ap = dataframe['{}_AP'.format(cardiovascular)].mean()
    cardiovascular_avg_hd = dataframe['{}_HD'.format(cardiovascular)].mean()
    print(f'{cardiovascular}: DSC = {cardiovascular_avg_dsc:.4f}, AP = {cardiovascular_avg_ap:.4f}, HD = {cardiovascular_avg_hd:.4f}')
print(20 * '<')