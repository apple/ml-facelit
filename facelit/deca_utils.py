import os
import zipfile
import torch 
import numpy as np 
from random import shuffle, randrange
from geometry_utils import get_extrinsics_from_axis_angle_and_cam
from tqdm import tqdm

class DecaSampler:
    def __init__(self, file_path, n_samples=10):
        self.file_path = file_path
        self.n_samples = n_samples
        self.samples = []
        self.load_samples_deca()
        self.n_samples = len(self.samples)

    def __getitem__(self, x):
        return self.samples[x]

    def load_samples_deca(self):
        label_zip = zipfile.ZipFile(self.file_path)
        label_files = [x for x in label_zip.namelist() if x.endswith('.pth')]
        print("loading deca light coefficients")
        label_files = sorted(label_files)
        for i, label_fname in enumerate(tqdm(label_files)):
            with label_zip.open(label_fname, 'r') as f:
                label_data = torch.load(f)
                mat_4x4 = get_extrinsics_from_axis_angle_and_cam(label_data['pose'][0][:3], label_data['cam'][0])
                cam_3x3 = np.array([4.2647, 0, 0.5, 0, 4.2647, 0.5, 0, 0, 1])
                cond = np.concatenate((mat_4x4.flatten(), cam_3x3, label_data['light'].flatten().numpy()))
                
                self.samples.append(cond)
            if i == self.n_samples:
                break
    
    def sample(self):
        return self.samples[randrange(self.n_samples)]
    
    def __len__(self):
        return len(self.samples)