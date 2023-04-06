from re import L
import click
import glob
import os 
import torch
from tqdm import tqdm
import numpy as np 
from geometry_utils import get_extrinsics_from_axis_angle_and_cam
from scipy.spatial.transform import Rotation
import torch.nn.functional as F

@click.command()
@click.option('--eval-dir', 'eval_dir', help='eval dir', required=True)
def calc_deca_consistency_metrics(eval_dir):
    pose_errors = []
    depth_errors = []
    light_errors = []

    label_files = sorted(glob.glob(os.path.join(eval_dir, 'label/*.pth')))
    decafit_files = sorted(glob.glob(os.path.join(eval_dir, 'deca_fits/*.pth')))

    label_files = label_files[:1024]
    decafit_files = decafit_files[:1024]

    assert len(label_files) == 1024

    for lf, df in tqdm(zip(label_files, decafit_files)):
        assert lf[-12:] == df[-12:]
        label = torch.load(lf)
        deca_fit = torch.load(df)

        label_rotmat = label['rotmat'][0,:3,:3].numpy()
        deca_extrinsic = get_extrinsics_from_axis_angle_and_cam(deca_fit['pose'][0,:3], deca_fit['cam'][0])
        deca_rotmat = deca_extrinsic[:3, :3]
        
        label_rotmat = Rotation.from_matrix(label_rotmat)
        deca_rotmat = Rotation.from_matrix(deca_rotmat)

        pose_err = label_rotmat.as_euler('xyz') - deca_rotmat.as_euler('xyz')

        # No angle difference should exceed np.pi
        pose_err = (pose_err + np.pi) % (2*np.pi) - np.pi
        pose_errors.append((pose_err**2).mean())

        light_err = (deca_fit['light'] - label['light']).pow(2).mean().item()
        light_errors.append(light_err)


    print(f"Pose Errors: {np.mean(pose_errors)}")
    print(f"Light Errors: {np.mean(light_errors)}")
    
    with open('eval_metrics.log', "a") as f:
        print(f"Img dir {eval_dir}, Pose_Errors: {np.mean(pose_errors)}, Light_Errors: {np.mean(light_errors)}", file=f)

def normalize_tensor(x):
    mean = x.mean()
    std = x.std()
    norm_x = (x - mean) / (std) # + 1e-8)
    return norm_x

if __name__ == '__main__':
    calc_deca_consistency_metrics()
