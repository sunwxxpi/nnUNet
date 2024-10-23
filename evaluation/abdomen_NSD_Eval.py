import os
import argparse
import pandas as pd
import numpy as np
import nibabel as nb
from collections import OrderedDict
from tqdm import tqdm
from SurfaceDice import compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient

def find_lower_upper_zbound(organ_mask):
    """
    Parameters
    ----------
    seg : TYPE
        DESCRIPTION.

    Returns
    -------
    z_lower: lower bound in z axis: int
    z_upper: upper bound in z axis: int

    """
    organ_mask = np.uint8(organ_mask)
    assert np.max(organ_mask) == 1, print('mask label error!')
    z_index = np.where(organ_mask > 0)[2]
    z_lower = np.min(z_index)
    z_upper = np.max(z_index)
    
    return z_lower, z_upper

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

gt_path = args.gt_path
seg_path = args.seg_path
save_path = args.save_path

filenames = os.listdir(seg_path)
filenames = [x for x in filenames if x.endswith('.nii.gz')]
filenames = [x for x in filenames if os.path.exists(join(seg_path, x))]
filenames.sort()

seg_metrics = OrderedDict()
seg_metrics['Name'] = list()
label_tolerance = OrderedDict({'Liver': 5, 'RK': 3, 'Spleen': 3, 'Pancreas': 5, 
                               'Aorta': 2, 'IVC': 2, 'RAG': 2, 'LAG': 2, 'Gallbladder': 2, 
                               'Esophagus': 3, 'Stomach': 5, 'Duodenum': 7, 'LK': 3})

for organ in label_tolerance.keys():
    seg_metrics['{}_NSD'.format(organ)] = list()

for name in tqdm(filenames):
    seg_metrics['Name'].append(name)
    # load grond truth and segmentation
    gt_nii = nb.load(join(gt_path, name))
    case_spacing = gt_nii.header.get_zooms()
    gt_data = np.uint8(gt_nii.get_fdata())
    seg_data = np.uint8(nb.load(join(seg_path, name)).get_fdata())

    for i, organ in enumerate(label_tolerance.keys(), 1):
        if np.sum(gt_data == i) == 0 and np.sum(seg_data == i) == 0:
            DSC_i = 1
            NSD_i = 1
        elif np.sum(gt_data == i) == 0 and np.sum(seg_data == i) > 0:
            DSC_i = 0
            NSD_i = 0
        else:
            if i == 5 or i == 6 or i == 10: # for Aorta, IVC, and Esophagus, only evaluate the labelled slices in ground truth
                z_lower, z_upper = find_lower_upper_zbound(gt_data == i)
                organ_i_gt = gt_data[:, :, z_lower:z_upper] == i
                organ_i_seg = seg_data[:, :, z_lower:z_upper] == i
            else:
                organ_i_gt = gt_data == i
                organ_i_seg = seg_data == i
            
            surface_distances = compute_surface_distances(organ_i_gt, organ_i_seg, case_spacing)
            NSD_i = compute_surface_dice_at_tolerance(surface_distances, label_tolerance[organ])
            
        seg_metrics['{}_NSD'.format(organ)].append(round(NSD_i, 4))

    dataframe = pd.DataFrame(seg_metrics)
    # dataframe.to_csv(seg_path + '_DSC.csv', index=False)
    dataframe.to_csv(save_path, index=False)

case_avg_NSD = dataframe.mean(axis=0, numeric_only=True)

print(20 * '>')
print(f'Average NSD for {basename(seg_path)}: {case_avg_NSD.mean()}')
print("\nOrgan-wise Average NSD: ")
for organ in label_tolerance.keys():
    organ_avg_dsc = dataframe['{}_NSD'.format(organ)].mean()
    print(f'{organ}: {organ_avg_dsc:.4f}')
print(20 * '<')