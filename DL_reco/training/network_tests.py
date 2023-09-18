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
        nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1),
        nn.ReLU()
    ]

    return nn.Sequential(*list_of_operations)

### Class for encoding part of the UNet. In other words, this is the part of
### the UNet which goes down with maxpooling.
class encoding(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        ### Defining instance variables
        self.in_channels = in_channels

        self.convs_and_relus1 = double_conv_and_ReLU(self.in_channels, out_channels=32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.convs_and_relus2 = double_conv_and_ReLU(in_channels=32, out_channels=64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.convs_and_relus3 = double_conv_and_ReLU(in_channels=64, out_channels=128)

    ### Must have forward function. Follows skip connecting UNet architechture
    def forward(self, g):
        g_start = g
        encoding_features = []
        g = self.convs_and_relus1(g)
        encoding_features.append(g)
        g = self.maxpool1(g)
        g = self.convs_and_relus2(g)
        encoding_features.append(g)
        g = self.maxpool2(g)
        g = self.convs_and_relus3(g)

        return g, encoding_features, g_start

### Class for decoding part of the UNet. This is the part of the UNet which
### goes back up with transpose of the convolution
class decoding(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        ### Defining instance variables
        self.out_channels = out_channels

        self.transpose1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2,2), stride=2, padding=0)
        self.convs_and_relus1 = double_conv_and_ReLU(in_channels=128, out_channels=64)
        self.transpose2 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2,2), stride=2, padding=0)
        self.convs_and_relus2 = double_conv_and_ReLU(in_channels=64, out_channels=32)
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=self.out_channels, kernel_size=(3,3), padding=1)

    ### Must have forward function. Follows skip connecting UNet architechture
    def forward(self, g, encoding_features, g_start):
        g = self.transpose1(g)
        # print('g', g.shape)
        g = torch.cat([g, encoding_features[-1]], dim=1)
        # print('encoding', encoding_features[-1].shape)
        # print('g', g.shape)
        encoding_features.pop()
        g = self.convs_and_relus1(g)
        g = self.transpose2(g)
        g = torch.cat([g, encoding_features[-1]], dim=1)
        encoding_features.pop()
        g = self.convs_and_relus2(g)
        g = self.final_conv(g)

        g = g_start + g

        return g

### Class for the UNet model itself
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = encoding(self.in_channels)
        self.decoder = decoding(self.out_channels)

    ### Must have forward function. Calling encoder and deoder classes here
    ### and making the whole UNet model
    def forward(self, g):
        g, encoding_features, g_start = self.encoder(g)
        g = self.decoder(g, encoding_features, g_start)

        return g
    


amount_of_images = 5

device = 'cpu'

# UNet = UNet(in_channels=1, out_channels=1).to(device)
# UNet.load_state_dict(torch.load('/scratch2/antti/networks/'+'UNet1_005.pth'))

# UNet.eval()

image = get_images('/scratch2/antti/summer2023/test_walnut/', amount_of_images, scale_number=2)
image = np.array(image, dtype='float32')
image = torch.from_numpy(image).float().to(device)

shape = (np.shape(image)[1], np.shape(image)[2])
domain, geometry, ray_transform, output_shape = geometry_and_ray_trafo(setup='full', shape=shape, device=device, factor_lines = 2)
fbp_operator = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_transform, padding=1)

### Using odl functions to make odl operators into PyTorch modules
ray_transform_module = OperatorModule(ray_transform).to(device)
adjoint_operator_module = OperatorModule(ray_transform.adjoint).to(device)
fbp_operator_module = OperatorModule(fbp_operator).to(device)

### Making sinograms from the images using Radon transform module
# sinograms = ray_transform_module(images)

sinograms = ray_transform_module(image) #.cpu().detach().numpy()

### Allocating used tensors
noisy_sinograms = torch.zeros((sinograms.shape[0], ) + output_shape).cpu().detach().numpy()
rec_images = torch.zeros((sinograms.shape[0], ) + shape)


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
    sinogram_k = sinograms[k,:,:].cpu().detach().numpy()
    # print(sinogram_k.std())
    noise = np.random.normal(mean, sinogram_k.std(), sinogram_k.shape) * percentage
    noisy_sinograms[k,:,:] = sinogram_k + noise

noisy_sinograms = np.array(noisy_sinograms, dtype='float32')
noisy_sinograms = torch.from_numpy(noisy_sinograms).float().to(device)

rec_images = fbp_operator_module(noisy_sinograms)
rec_images = rec_images[:,None,:,:]
print(rec_images.shape)


UNet = UNet(in_channels=1, out_channels=1).to(device)
UNet.load_state_dict(torch.load('/scratch2/antti/networks/'+'UNet1_005.pth', map_location=device))

UNet.eval()

reco = UNet(rec_images)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(rec_images[0,0,:,:].cpu().detach().numpy())
plt.subplot(1,3,2)
plt.imshow(reco[0,0,:,:].cpu().detach().numpy())
plt.subplot(1,3,3)
plt.imshow(image[0,:,:].cpu().detach().numpy())
plt.show()



# for k in range(amount_of_images):
#     #coeff = np.max(np.max(sinograms[k,:,:].cpu().detach().numpy()))
#     # mean = 0.05 #* coeff
#     # variance = 0.01 #* coeff
#     # sigma = variance ** 0.5
#     image = image.cpu().detach().numpy()
#     print(image.std())
#     noise = np.random.normal(mean, image.std(), image.shape) * percentage
#     noisy_image = image + noise




