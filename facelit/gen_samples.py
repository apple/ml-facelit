# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile
import random

import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics, GaussianCameraPoseSampler
from deca_utils import DecaSampler
from light_utils import LightSampler, paste_light_on_img_tensor, rotate_SH_coeffs, angle_in_a_circle
from torch_utils import misc
from training.triplane import TriPlaneGenerator


#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

    
#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--dataset', 'dataset', type=click.Choice(['ffhq', 'celeba_hq', 'metfaces']), default='ffhq')
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--shapes', help='Export shapes as .mrc files viewable in ChimeraX', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--double-views', help='Render double views for FaceID consistency', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--deca-samples', help='Render views for deca pose depth light consistency', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--relight', help='Render relighted', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--shape-res', help='', type=int, required=False, metavar='int', default=512, show_default=True)
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--shape-format', help='Shape Format', type=click.Choice(['.mrc', '.ply']), default='.mrc')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    dataset: str,
    shapes: bool,
    shape_res: int,
    fov_deg: float,
    shape_format: str,
    class_idx: Optional[int],
    reload_modules: bool,
    double_views: bool,
    deca_samples: bool,
    relight: bool,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate an image using pre-trained FFHQ model.
    python gen_samples.py --outdir=output --trunc=0.7 --seeds=0-5 --shapes=True\\
        --network=ffhq-rebalanced-128.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new
    os.makedirs(outdir, exist_ok=True)

    cam2world_pose = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device)
    intrinsics = FOV_to_intrinsics(fov_deg, device=device)

    if dataset == 'ffhq':
        deca_fits_path = '../data/FFHQ2x_512_deca_fits.zip'
        h_stddev = 0.15
        v_stddev = 0.18
    elif dataset == 'metfaces':
        deca_fits_path = '../data/MetFaces_512_deca_fits.zip'
        h_stddev = 0.08
        v_stddev = 0.15
    elif dataset == 'celeba_hq':
        deca_fits_path = '../data/CelebA_HQ_512_deca_fits.zip'
        h_stddev = 0.12
        v_stddev = 0.11


    light_sampler = LightSampler(file_path=deca_fits_path, n_samples=1048)

    if deca_samples:
        deca_sampler = DecaSampler(file_path=deca_fits_path, n_samples=1048)
        os.makedirs(f'{outdir}/img', exist_ok=True)
        os.makedirs(f'{outdir}/viz', exist_ok=True)
        os.makedirs(f'{outdir}/label', exist_ok=True)
        G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * 2)
        G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * 2)

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        imgs = []

        if deca_samples:
            conditioning_params = deca_sampler.sample() #[seed_idx]
            conditioning_params = torch.Tensor(conditioning_params).reshape(-1, 52).to(device)
            light_params = conditioning_params[:,25:]

            ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            out = G.synthesis(ws, conditioning_params)
            img = out['image'][:, :3]
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/img/seed{seed:04d}.png')

            sphere_size = 128
            img_sphr = paste_light_on_img_tensor(sphere_size, light_params.reshape(9,3), out['image'][:, :3])
            img_sphr = (img_sphr.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img_sphr[0].cpu().numpy(), 'RGB').save(f'{outdir}/viz/seed{seed:04d}.png')

            label_dict = {}
            label_dict['image_depth'] = out['image_depth'].cpu()
            label_dict['rotmat'] = conditioning_params[:, :16].reshape(-1, 4, 4).cpu()
            label_dict['light'] = light_params.reshape(-1, 9, 3).cpu()

            torch.save(label_dict, f'{outdir}/label/seed{seed:04d}.pth')


        elif double_views:
            angles = [(np.random.normal(0, h_stddev), np.random.normal(0, v_stddev))]
            angles += [(np.random.normal(0, h_stddev), np.random.normal(0, v_stddev))]

            light_center = torch.Tensor(light_sampler.load_deca_center_light()).to(device)
            for angle_y, angle_p in angles:
                cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivot', [0, 0, 0]), device=device)
                cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9), light_center.reshape(-1, 27)], 1)
                conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9), light_center.reshape(-1, 27)], 1)
        
                ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                img = G.synthesis(ws, camera_params)['image']
                if img.shape[1] == 6:
                    img = img[:,:3,:,:]

                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                imgs.append(img)

            PIL.Image.fromarray(imgs[0][0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}_view1.png')
            PIL.Image.fromarray(imgs[1][0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}_view2.png')

        else:
            angle_p = -0.2
            light_center = torch.Tensor(light_sampler[0]).to(device)
            for angle_y, angle_p in [(.3, angle_p), (0, angle_p), (-.3, angle_p)]:
                cam_pivot = torch.tensor([0, 0, 0.2], device=device)
                cam_radius = G.rendering_kwargs.get('avg_camera_radius', 2.7)
                cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=cam_radius, device=device)
                conditioning_cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=cam_radius, device=device)
                camera_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9), light_center.reshape(-1, 27)], 1)
                conditioning_params = torch.cat([conditioning_cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9), light_center.reshape(-1, 27)], 1)

                ws = G.mapping(z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
                img = G.synthesis(ws, camera_params)['image']
                if img.shape[1] == 6:
                    img = img[:,:3,:,:]
                
                if relight:
                    sphere_size = 128
                    img = paste_light_on_img_tensor(sphere_size, light_center, img)
                    lights_angles = [angle_in_a_circle(x, axis=ax) for (x, ax) in [(0.3, 'z'), (0.5, 'x'), (0.9, 'z')]]
                    light_strength = [1., 1., 1.]
                    for light_ang, amp in zip(lights_angles, light_strength):
                        new_light = amp*torch.Tensor(rotate_SH_coeffs(light_center, light_ang)).to(device)
                        new_cond_params = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9), new_light.reshape(-1, 27)], 1)
                        new_img = G.synthesis(ws, c=new_cond_params, noise_mode='const')['image']
                        if new_img.shape[1] == 6:
                            new_img = new_img[:,:3,:,:]

                        new_img = paste_light_on_img_tensor(sphere_size, new_light, new_img)
                        img = torch.cat((img, new_img), dim=2)

                img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                imgs.append(img)

            img = torch.cat(imgs, dim=2)

            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')

        if shapes:
            # extract a shape.mrc with marching cubes. You can view the .mrc file using ChimeraX from UCSF.
            max_batch=1000000

            samples, voxel_origin, voxel_size = create_samples(N=shape_res, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'] * 1)#.reshape(1, -1, 3)
            samples = samples.to(z.device)
            sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=z.device)
            transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=z.device)
            transformed_ray_directions_expanded[..., -1] = -1

            head = 0
            with tqdm(total = samples.shape[1]) as pbar:
                with torch.no_grad():
                    while head < samples.shape[1]:
                        torch.manual_seed(0)
                        sigma = G.sample(samples[:, head:head+max_batch], transformed_ray_directions_expanded[:, :samples.shape[1]-head], z, conditioning_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, noise_mode='const')['sigma']
                        sigmas[:, head:head+max_batch] = sigma
                        head += max_batch
                        pbar.update(max_batch)

            sigmas = sigmas.reshape((shape_res, shape_res, shape_res)).cpu().numpy()
            sigmas = np.flip(sigmas, 0)

            # Trim the border of the extracted cube
            pad = int(30 * shape_res / 256)
            pad_value = -1000
            sigmas[:pad] = pad_value
            sigmas[-pad:] = pad_value
            sigmas[:, :pad] = pad_value
            sigmas[:, -pad:] = pad_value
            sigmas[:, :, :pad] = pad_value
            sigmas[:, :, -pad:] = pad_value

            if shape_format == '.ply':
                from shape_utils import convert_sdf_samples_to_ply
                convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'seed{seed:04d}.ply'), 
                    level=10) #, rot=conditioning_params[:,:16].reshape(4,4).cpu().numpy())
            elif shape_format == '.mrc': # output mrc
                with mrcfile.new_mmap(os.path.join(outdir, f'seed{seed:04d}.mrc'), overwrite=True, shape=sigmas.shape, mrc_mode=2) as mrc:
                    mrc.data[:] = sigmas


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
