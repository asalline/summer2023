import cv2 as cv
import torch
import odl
import os
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from odl.contrib.torch import OperatorModule


def get_images(path, amount_of_images='all', scale_number=1):

    all_images = []
    all_image_names = os.listdir(path)
    print(len(all_image_names))
    if amount_of_images == 'all':
        for name in all_image_names:
            temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
            image = temp_image[90:410, 90:410]
            image = image[0:320:scale_number, 0:320:scale_number]
            image = image / 0.07584485627272729
            all_images.append(image)
    else:
        temp_indexing = np.random.permutation(len(all_image_names))[:amount_of_images]
        
# =============================================================================
#         images_to_take = all_image_names[temp_indexing]
# =============================================================================
        images_to_take = [all_image_names[i] for i in temp_indexing]
        for name in images_to_take:
            temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
            image = temp_image[90:410, 90:410]
            image = image[0:320:scale_number, 0:320:scale_number]
            image = image / 0.07584485627272729
            all_images.append(image)
# =============================================================================
#         image = temp_image[0:np.shape(temp_image)[0]:scale_number, 0:np.shape(temp_image)[1]:scale_number]
#         #print(image.shape)
#         new_image = np.zeros((128, 128), dtype=image.dtype)
#         new_image[:image.shape[0], :image.shape[1]] = image
#         all_images.append(new_image)
# =============================================================================
    
    return all_images

def geometry_and_ray_trafo(setup='full', min_domain_corner=[-1,-1], max_domain_corner=[1,1], \
                           shape=(100,100), source_radius=2, detector_radius=1, \
                           dtype='float32', device='cpu', factor_lines = 1):

    device = 'astra_' + device
    print(device)
    domain = odl.uniform_discr(min_domain_corner, max_domain_corner, shape, dtype=dtype)

    if setup == 'full':
        angles = odl.uniform_partition(0, 2*np.pi, 360)
        lines = odl.uniform_partition(-1*np.pi, np.pi, int(1028/factor_lines))
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (360, int(1028/factor_lines))
    elif setup == 'sparse':
        angle_measurements = 100
        line_measurements = int(512/factor_lines)
        angles = odl.uniform_partition(0, 2*np.pi, angle_measurements)
        lines = odl.uniform_partition(-1*np.pi, np.pi, line_measurements)
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (angle_measurements, line_measurements)
    elif setup == 'limited':
        starting_angle = 0
        final_angle = np.pi * 3/4
        angles = odl.uniform_partition(starting_angle, final_angle, 360)
        lines = odl.uniform_partition(-1*np.pi, np.pi, int(512/factor_lines))
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (int(360), int(512/factor_lines))
        
    # domain = odl.uniform_discr(min_domain_corner, max_domain_corner, output_shape, dtype=dtype)

    ray_transform = odl.tomo.RayTransform(domain, geometry, impl=device)

    return domain, geometry, ray_transform, output_shape


### Function needed when defining the UNet encoding and decoding parts
def double_conv_and_ReLU(in_channels, out_channels):
    list_of_operations = [
        nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1),
        nn.ReLU()
    ]

    return nn.Sequential(*list_of_operations)

class LPD_step(nn.Module):
    def __init__(self, operator, adjoint_operator, operator_norm, device='cuda'):
        super().__init__()
        
        self.operator = operator
        self.adjoint_operator = adjoint_operator
        self.operator_norm = operator_norm
        self.device = device
        
        
        self.primal_step = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3,3), padding=1)
            )

        self.dual_step = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3,3), padding=1)
            )

        self.to(device)
        
    def forward(self, f_rec_images, g_sinograms, h):
        
        
        f_sinogram = self.operator(f_rec_images) / self.operator_norm
        u = torch.cat([h, f_sinogram, g_sinograms / self.operator_norm], dim=1)
        h = h + self.dual_step(u)
        
        adjoint_eval = self.adjoint_operator(h) / self.operator_norm
        u = torch.cat([f_rec_images, adjoint_eval], dim=1)
        f_rec_images = f_rec_images + self.primal_step(u)
        
        return f_rec_images, h
        
class LPD(nn.Module):
    def __init__(self, operator, adjoint_operator, operator_norm, n_iter, device='cuda'):
        super().__init__()
        
        self.operator = operator
        self.adjoint_operator = adjoint_operator
        self.operator_norm = operator_norm
        self.n_iter = n_iter
        self.device = device
        
        for k in range(self.n_iter):
            step = LPD_step(operator, adjoint_operator, operator_norm)
            setattr(self, f'step{k}', step)
            
    def forward(self, f_rec_images, g_sinograms):
        
        h = torch.zeros(g_sinograms.shape).to(self.device)
        
        for k in range(self.n_iter):
            step = getattr(self, f'step{k}')
            f_rec_images, h = step(f_rec_images, g_sinograms, h)
            
        return f_rec_images

amount_of_images = 1

device = 'cuda'

image = get_images('/scratch2/antti/summer2023/test_walnut/', amount_of_images=5, scale_number=2)
image = np.array(image, dtype='float32')
image = torch.from_numpy(image).float().to(device)


shape = (np.shape(image)[1], np.shape(image)[2])
domain, geometry, ray_transform, output_shape = geometry_and_ray_trafo(setup='full', shape=shape, device=device, factor_lines = 2)
fbp_operator = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_transform, padding=1)

operator_norm = odl.power_method_opnorm(ray_transform)

### Using odl functions to make odl operators into PyTorch modules
ray_transform_module = OperatorModule(ray_transform).to(device)
adjoint_operator_module = OperatorModule(ray_transform.adjoint).to(device)
fbp_operator_module = OperatorModule(fbp_operator).to(device)



### Making sinograms from the images using Radon transform module
# sinograms = ray_transform_module(images)

sinogram = ray_transform_module(image) #.cpu().detach().numpy()

### Allocating used tensors
noisy_sinograms = torch.zeros((sinogram.shape[0], ) + output_shape)
rec_images = torch.zeros((sinogram.shape[0], ) + shape)


mean = 0
variance = 0.005
sigma = variance ** 0.5
percentage = 0.05

### Adding Gaussian noise to the sinograms. Here some problem solving is
### needed to make this smoother.
for k in range(amount_of_images):
    #coeff = np.max(np.max(sinograms[k,:,:].cpu().detach().numpy()))
    # mean = 0.05 #* coeff
    # variance = 0.01 #* coeff
    # sigma = variance ** 0.5
    sinogram = sinogram.cpu().detach().numpy()
    print(sinogram.std())
    noise = np.random.normal(mean, sinogram.std(), sinogram.shape) * percentage
    noisy_sinogram = sinogram + noise

noisy_sinogram = np.array(noisy_sinogram, dtype='float32')
noisy_sinogram = torch.from_numpy(noisy_sinogram).float().to(device)

noisy_image = fbp_operator_module(noisy_sinogram)

sinogram = torch.from_numpy(sinogram).float().to(device)
noisy_image = noisy_image[:,None,:,:]
noisy_sinogram = noisy_sinogram[:,None,:,:]
sinogram = sinogram[:,None,:,:]
print(noisy_image.shape)
print(noisy_sinogram.shape)

LPD = LPD(ray_transform_module, adjoint_operator_module, operator_norm, n_iter=10, device=device)
LPD.load_state_dict(torch.load('/scratch2/antti/networks/'+'LPD1_005.pth'))

LPD.eval()


reco = LPD(noisy_image[:,:,:], noisy_sinogram[:,:,:])

loss_test = nn.MSELoss()

def psnr(loss):
    
    psnr = 10 * np.log10(1.0 / loss+1e-10)
    
    return psnr

MSE = loss_test(image[0,:,:], reco[0,0,:,:]).cpu().detach().numpy()
print(f'{MSE:.2e}')
print(f'{psnr(MSE):.2f}')

# print('reco', type(reco))

plt.figure()
plt.subplot(1,3,1)
plt.imshow(noisy_image[0,0,:,:].cpu().detach().numpy())
plt.subplot(1,3,2)
plt.imshow(reco[0,0,:,:].cpu().detach().numpy())
plt.subplot(1,3,3)
plt.imshow(image[0,:,:].cpu().detach().numpy())
# plt.colorbar()
plt.show()














