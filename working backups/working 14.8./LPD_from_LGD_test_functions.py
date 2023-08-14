### Necessary functions needed for "FILENAME HERE" to work.
### Author: Antti SÃ¤llinen
### Date: July 2023

### Needed packages: -odl
###                  -PyTorch
###                  -NumPy
###                  -os
###                  -OpenCv
###


### Importing packages
import os
import cv2 as cv
import numpy as np
import odl
import torch
import torch.nn as nn


### Function that takes all the images from the path directory and crops them.
### Cropping part is hardcoded to match certain type of pictures. One probably
### needs to change the values.
### Inputs: -'path': path to directory where the images are
###         -'amount_of_images': how many images one wants to get from the
###                              given directory
###         -'scale_number': number of how many pixels does the function skip.
###                          Eg. scale_number = 4 -> every 4th pixel is taken
###                          from the original image
### Outputs: -'all_images': list of all images taken from directory
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


### Function that defines mathematical background for the script.
### More precise function defines geometry for the Radon transform.
### Inputs: -'setup': determines what kind of geometry one wants. Possible
###                   choices are 'full', 'sparse', 'limited'. Default: 'full'
###         -'min_domain_corner': Determines where the bottom left corner is
###                               is in the geometry. Default: [-1,-1]
###         -'max_domain_corner': Determines where the upper right corner is
###                               is in the geometry. Default: [1,1]
###         -'shape': how many points there is in x- and y-axis between the
###                   corners of the geometry. Default: (100,100)
###         -'source_radius': radius of the 'object' when taken measurements.
###                           Default: 2
###         -'detector_radius': radius of the ??? when taken measurements.
###                             Default: 1
###         -'dtype': Python data type. Default: 'float32'
###         -'device': Device which is used in calculations. Default: 'cpu'
###         -'factor_lines': Parameter which controls the line-measurements
###                          in 'sparse' and 'limited' geometries.
###                          Default: 1
### Outputs: -'domain': odl domain, not really used, could be deleted from
###                     the outputs
###          -'geometry': odl geometry, could be deleted from the outputs
###          -'ray_transform': Radon transform operator defined by
###                            given geometry
###          -'output_shape': Shape defined by angles and lines in geometry.
###                           Needed in the allocations.
def geometry_and_ray_trafo(setup='full', min_domain_corner=[-1,-1], max_domain_corner=[1,1], \
                           shape=(100,100), source_radius=2, detector_radius=1, \
                           dtype='float32', device='cpu', factor_lines = 1):

    device = 'astra_' + device
    print(device)
    domain = odl.uniform_discr(min_domain_corner, max_domain_corner, shape, dtype=dtype)

    if setup == 'full':
        angles = odl.uniform_partition(0, 2*np.pi, 100)
        lines = odl.uniform_partition(-1*np.pi, np.pi, int(100))
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (100,100)
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

### Class for the Learned Gradient Descent (LGD) algorithm.
class LGD(nn.Module):
    def __init__(self, adjoint_operator_module, operator_module, in_channels_dual, \
                 out_channels_dual, in_channels_primal, out_channels_primal, n_iter, \
                     device='cuda'):
        super().__init__()

        ### Defining instance variables
        self.in_channels_dual = in_channels_dual
        self.out_channels_dual = out_channels_dual
        self.in_channels_primal = in_channels_primal
        self.out_channels_primal = out_channels_primal
        # self.step_length = step_length
        self.step_length = nn.Parameter(torch.zeros(1,1,1,1))
        self.n_iter = n_iter
        self.device = device
        
        self.forward_operator = operator_module
        self.adjoint_operator = adjoint_operator_module
        
        Dual_layer = [
            double_conv_and_ReLU(in_channels=self.in_channels_dual, out_channels=32),
            nn.Conv2d(in_channels=32, out_channels=self.out_channels_dual, kernel_size=(3,3), padding=1)
        ]
        
        self.Dual_layers = nn.Sequential(*Dual_layer)
        self.Dual_layers2 = [self.Dual_layers for k in range(self.n_iter)]
        
        Primal_layer = [
            double_conv_and_ReLU(in_channels=self.in_channels_primal, out_channels=32),
            nn.Conv2d(in_channels=32, out_channels=self.out_channels_primal, kernel_size=(3,3), padding=1)
        ]
        
        self.Primal_layers = nn.Sequential(*Primal_layer)
        self.Primal_layers2 = [self.Primal_layers for k in range(self.n_iter)]
            
        self.to(device)
        # self.conv1 = nn.Conv2d(in_channels=self.in_channels, \
        #                         out_channels=32, kernel_size=(3,3), padding=1)
        # self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1)
        # self.relu2 = nn.ReLU()
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1)
        # self.relu3 = nn.ReLU()
        # self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1)
        # self.relu4 = nn.ReLU()
        # self.conv5 = nn.Conv2d(in_channels=32, out_channels=self.out_channels, \
        #                         kernel_size=(3,3), padding=1)
        # self.relu5 = nn.ReLU()
        
    def forward(self, f_rec_images, g_sinograms):
        
        h = torch.zeros(g_sinograms.shape).to(self.device)
        
        for i in range(self.n_iter):
            # print('1')
            f_sinogram = self.forward_operator(f_rec_images)
            # dual layer
            u = torch.cat([h, f_sinogram, g_sinograms], dim=1)
            h = h + self.Dual_layers2[i](u)
            # print('2')
            # primal layer
            adjoint_eval = self.adjoint_operator(h) # (output of dual - g_sinograms)
            u = torch.cat((adjoint_eval, f_rec_images), dim=1)
            f_rec_images = f_rec_images + self.Primal_layers2[i](u)
            # print('3')
        
        return f_rec_images #self.step_length


