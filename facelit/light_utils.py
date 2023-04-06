import os
import zipfile
import torch 
import numpy as np 
from random import shuffle, randrange
import scipy.special
from pyshtools.rotate import SHRotateRealCoef, djpi2
class LightSampler:
    def __init__(self, file_path, n_samples=10):
        self.file_path = file_path
        self.n_samples = n_samples
        self.samples = []
        self.load_samples_deca()

    def __getitem__(self, x):
        return self.samples[x]

    def load_samples_deca(self):
        label_zip = zipfile.ZipFile(self.file_path)
        label_files = [x for x in label_zip.namelist() if x.endswith('.pth')]
        print("loading deca light coefficients")
        label_files = sorted(label_files)
        for i, label_fname in enumerate(label_files):
            with label_zip.open(label_fname, 'r') as f:
                label_data = torch.load(f)
                self.samples.append(label_data['light'].squeeze().numpy())
            if i == self.n_samples:
                break
    
    def load_deca_center_light(self):
        light_center = np.array([[ 3.2057941 ,  3.19894958,  3.20620155],
            [-0.2028061 , -0.21840217, -0.20184311],
            [ 0.14086065,  0.14567219,  0.1473848 ],
            [-0.00488449, -0.00591755, -0.01403105],
            [-0.04424239, -0.04043309, -0.04472703],
            [-0.15366814, -0.16222024, -0.16256415],
            [-0.16824984, -0.16911056, -0.17083314],
            [-0.26489747, -0.2669786 , -0.27070296],
            [-0.14320635, -0.14578062, -0.14734463]])
        return light_center
    
    def sample(self):
        return self.samples[randrange(self.n_samples)]
    
    def __len__(self):
        return len(self.samples)

def get_normals(img_size):
    '''
        https://github.com/zhhoper/DPR/blob/master/testNetwork_demo_512.py
        
    '''
    x = np.linspace(-1, 1, img_size)
    z = np.linspace(1, -1, img_size)
    x, z = np.meshgrid(x, z)

    mag = np.sqrt(x**2 + z**2)
    valid = mag <=1
    y = -np.sqrt(1 - (x*valid)**2 - (z*valid)**2)
    x = x * valid
    y = y * valid
    z = z * valid
    normal = np.concatenate((x[...,None], y[...,None], z[...,None]), axis=2)
    normal = np.reshape(normal, (-1, 3))
    return normal, valid

def get_shading(normal, SH):
    '''
        https://github.com/zhhoper/DPR/blob/master/utils/utils_SH.py
        get shading based on normals and SH
        normal is Nx3 matrix
        SH: 9 x m vector
        return Nxm vector, where m is the number of returned images
    '''
    sh_basis = SH_basis(normal)
    shading = np.matmul(sh_basis, SH)
    #shading = np.matmul(np.reshape(sh_basis, (-1, 9)), SH)
    #shading = np.reshape(shading, normal.shape[0:2])
    return shading

def SH_basis(normal):
    '''
        https://github.com/zhhoper/DPR/blob/master/utils/utils_SH.py
        get SH basis based on normal
        normal is a Nx3 matrix
        return a Nx9 matrix
        The order of SH here is:
        1, Y, Z, X, YX, YZ, 3Z^2-1, XZ, X^2-y^2
    '''
    numElem = normal.shape[0]

    norm_X = normal[:,0]
    norm_Y = normal[:,1]
    norm_Z = normal[:,2]

    sh_basis = np.zeros((numElem, 9))
    att= np.pi*np.array([1, 2.0/3.0, 1/4.0])
    sh_basis[:,0] = 0.5/np.sqrt(np.pi)*att[0]

    sh_basis[:,1] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Y*att[1]
    sh_basis[:,2] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_Z*att[1]
    sh_basis[:,3] = np.sqrt(3)/2/np.sqrt(np.pi)*norm_X*att[1]

    sh_basis[:,4] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_X*att[2]
    sh_basis[:,5] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_Y*norm_Z*att[2]
    sh_basis[:,6] = np.sqrt(5)/4/np.sqrt(np.pi)*(3*norm_Z**2-1)*att[2]
    sh_basis[:,7] = np.sqrt(15)/2/np.sqrt(np.pi)*norm_X*norm_Z*att[2]
    sh_basis[:,8] = np.sqrt(15)/4/np.sqrt(np.pi)*(norm_X**2-norm_Y**2)*att[2]
    return sh_basis

def render_half_sphere(sh, img_size):
    '''
        sh: np.array (9x3)
        https://github.com/zhhoper/DPR/blob/master/testNetwork_demo_512.py
    '''
    sh = rotate_SH_coeffs(sh, np.array([np.pi/2, 0, 0]))

    normal, valid = get_normals(img_size)
    shading = get_shading(normal, sh)
    value = np.percentile(shading, 95)
    ind = shading > value
    shading[ind] = value
    shading = (shading - np.min(shading))/(np.max(shading) - np.min(shading))
    shading = (shading *255.0).astype(np.uint8)
    shading = np.reshape(shading, (img_size, img_size, 3))
    shading = shading * valid[:,:,None]
    return shading, valid

def shtools_matrix2vec(SH_matrix):
    '''
        for the sh matrix created by sh tools, 
        we create the vector of the sh
    '''
    numOrder = SH_matrix.shape[1]
    vec_SH = np.zeros(numOrder**2)
    count = 0
    for i in range(numOrder):
        for j in range(i,0,-1):
            vec_SH[count] = SH_matrix[1,i,j]
            count = count + 1
        for j in range(0,i+1):
            vec_SH[count]= SH_matrix[0, i,j]
            count = count + 1
    return vec_SH

def shtools_sh2matrix(coefficients, degree):
    '''
        convert vector of sh to matrix
    '''
    coeffs_matrix = np.zeros((2, degree + 1, degree + 1))
    current_zero_index = 0
    for l in range(0, degree + 1):
        coeffs_matrix[0, l, 0] = coefficients[current_zero_index]
        for m in range(1, l + 1):
            coeffs_matrix[0, l, m] = coefficients[current_zero_index + m]
            coeffs_matrix[1, l, m] = coefficients[current_zero_index - m]
        current_zero_index += 2*(l+1)
    return coeffs_matrix 
    
def rotate_SH_coeffs(sh, angles, dj=None):
    if dj is None:
        dj = djpi2(2)
    rotated = np.zeros(sh.shape)
    for i in range(sh.shape[1]):
        rotmat = SHRotateRealCoef(shtools_sh2matrix(sh[:,i], 2), angles, dj)
        rotated[:,i] = shtools_matrix2vec(rotmat)
    return rotated
    

def paste_light_on_img_tensor(sphere_size, light_coeff, img):
    '''
        sphere_size: int, denoting SxS sized half sphere 
        light_coeff: 9x3 tensor of sh coefficient
        img: BxCxHxW batched images
    '''

    device = img.device
    sphere_img, alpha_mask = render_half_sphere(light_coeff.cpu().numpy(), sphere_size)
    sphere_img = torch.Tensor(sphere_img).permute(2,0,1).to(device)
    sphere_img = (sphere_img - sphere_img.min()) / (sphere_img.max() - sphere_img.min()) * 2 - 1
    alpha_mask = torch.Tensor(alpha_mask).to(device)
    img[:,:,-sphere_size:,-sphere_size:] = (1 - alpha_mask[None,None,:,:]) * img[:,:,-sphere_size:,-sphere_size:] + alpha_mask[None,None,:,:] * sphere_img.unsqueeze(0)
    return img

def angle_in_a_circle(param, axis='z'):
    assert 0 <= param <= 1

    if axis == 'x':
        return np.array([np.pi/2, 0, param*2*np.pi])
    if axis == 'y':
        return np.array([np.pi/2, param*2*np.pi, 0])
    if axis == 'z':
        return np.array([param*2*np.pi, np.pi/2, 0])
    if axis == 'p':
        return np.array([param*2*np.pi, 0, 0])
