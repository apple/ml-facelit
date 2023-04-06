# This file is adapted from
# 
# Xiaoming Zhao, Fangchang Ma, David Güera, Zhile Ren, Alexander G. Schwing, 
# and Alex Colburn. Generative Multiplane Images: Making a 2D GAN 3D-Aware. ECCV 2022.
# https://github.com/apple/ml-gmpi

import argparse
import glob
import json
import os

import joblib
import numpy as np
import scipy.io as sio
import torch
import tqdm

from geometry_utils import get_extrinsics_from_axis_angle_and_cam
from scipy.spatial.transform import Rotation

SPHERE_CENTER = 1.0
SPHERE_R = 1.0
SPHERE_CENTER_VEC = torch.FloatTensor([0, 0, SPHERE_CENTER])


def normalize_vec(vec):
    mean = np.mean(vec)
    std = np.std(vec)
    norm_vec = (vec - mean) / (std + 1e-8)
    return norm_vec


def compute_depth_err(alinged_depth_f, pred_depth_f, pred_mask_f):

    depth = np.load(alinged_depth_f)

    pred_depth = np.load(pred_depth_f)
    pred_mask = np.load(pred_mask_f)

    pred_mask[depth < 1e-8] = 0

    # [#pixels, ]
    valid_rows, valid_cols = np.where(pred_mask == 1)

    pred_depth_pixs = pred_depth[valid_rows, valid_cols]
    depth_pixs = depth[valid_rows, valid_cols]

    norm_pred_depth = normalize_vec(pred_depth_pixs)
    norm_depth = normalize_vec(depth_pixs)

    err = np.mean(np.square(norm_pred_depth - norm_depth))

    return err



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--geo_dir", type=str, required=True)
    args = parser.parse_args()

    all_fs = sorted(list(glob.glob(os.path.join(args.geo_dir, f"img/detections/seed*.txt"))))[:1024]

    assert len(all_fs) == 1024

    depth_err_dict = {}
    angle_err_dict = {}
    err_mat = np.zeros((3,))
    for tmp_f in tqdm.tqdm(all_fs):
        i = os.path.basename(tmp_f).split(".")[0]
        tmp_aligned_depth_f = os.path.join(args.geo_dir, f"img/recon/aligned_depth/{i}.npy")
        tmp_pred_depth_f = os.path.join(args.geo_dir, f"img/recon/pred_depth/{i}.npy")
        tmp_pred_mask_f = os.path.join(args.geo_dir, f"img/recon/pred_mask/{i}.npy")
        tmp_pred_coeff_f = os.path.join(args.geo_dir, f"img/recon/coeffs/{i}.mat")
        
        tmp_depth_err = compute_depth_err(tmp_aligned_depth_f, tmp_pred_depth_f, tmp_pred_mask_f)
        depth_err_dict[i] = tmp_depth_err

    depth_err = np.mean(list(depth_err_dict.values()))
    depth_err_std = np.std(list(depth_err_dict.values()))
    print("\n", args.geo_dir, "\n")
    print("\ndepth: ", depth_err, depth_err_std, "\n")

    save_dict = {
        "depth": str(depth_err),
        "depth_std": str(depth_err_std),
    }

    with open('eval_metrics.log', 'a') as f:
        print(f"Img dir {args.geo_dir}, Depth Error: {depth_err} +/- {depth_err_std} ", file=f)

