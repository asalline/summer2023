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
            # image = image / 0.07584485627272729
            all_images.append(image)
    else:
        temp_indexing = np.random.permutation(len(all_image_names))[:amount_of_images]
        images_to_take = [all_image_names[i] for i in temp_indexing]
        for name in images_to_take:
            temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
            image = temp_image[90:410, 90:410]
            image = image[0:320:scale_number, 0:320:scale_number]
            # image = image / 0.07584485627272729
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
        angles = odl.uniform_partition(0, 2*np.pi, 100) #360
        lines = odl.uniform_partition(-1*np.pi, np.pi, 100) #int(1024/factor_lines)
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (100,100) #(360,int(1024/factor_lines)) #shape
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
def double_conv_and_PReLU(in_channels, out_channels):
    list_of_operations = [
        nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1),
        nn.PReLU(num_parameters=out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1),
        nn.PReLU(num_parameters=out_channels)
    ]

    return nn.Sequential(*list_of_operations)


class Dual_map(nn.Module):
    def __init__(self, in_channels, out_channels, device='cuda'):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        
        ### Defining dual layers
        dual_layers = [
            double_conv_and_PReLU(in_channels=self.in_channels, out_channels=32),
            nn.Conv2d(in_channels=32, out_channels=self.out_channels, kernel_size=(3,3), padding=1)
        ]

        self.dual_layers = nn.Sequential(*dual_layers)
        self.to(device)

    def forward(self, h, f, g, forward_operator):
        
        eval_f = forward_operator(f)
        # print('h', h.shape)
        # print('f', eval_f.shape)
        # print('g', g.shape)
        u = torch.cat((h, eval_f, g), dim=1)
        # print('u type', type(u))
        u = u.type(torch.float32)
        h = h + self.dual_layers(u)

        return h


class Primal_map(nn.Module):
    def __init__(self, in_channels, out_channels, device='cuda'):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        
        ### Defining primal layers
        primal_layers = [
            double_conv_and_PReLU(in_channels=self.in_channels, out_channels=32),
            nn.Conv2d(in_channels=32, out_channels=self.out_channels, kernel_size=(3,3), padding=1)
        ]

        self.primal_layers = nn.Sequential(*primal_layers)
        self.to(device)
        
    def forward(self, h, f, adjoint_operator):
        
        eval_h = adjoint_operator(h)
        u = torch.cat((eval_h, f), dim=1)
        f = f + self.primal_layers(u)
        
        return f
    
class LPD(nn.Module):
    def __init__(self, forward_operator, adjoint_operator, primal_map, dual_map, \
                 primal_shape, dual_shape, iterations=10, device='cuda'):
        super(LPD, self).__init__()
        
        self.forward_operator = forward_operator
        self.adjoint_operator = adjoint_operator
        self.primal_map = nn.ModuleList(primal_map)
        # self.primal_map = primal_map
        self.dual_map = nn.ModuleList(dual_map)
        # self.dual_map = dual_map
        self.primal_shape = primal_shape
        self.dual_shape = dual_shape
        self.iterations = iterations
        self.device = device
        
    def forward(self, g):
        
        f = torch.zeros((g.shape[0], ) + self.primal_shape).to(self.device)
        h = torch.zeros((g.shape[0], ) + self.dual_shape).to(self.device)
        # f = torch.zeros((1, 6) + self.primal_shape).to(self.device)
        # h = torch.zeros((1, 7) + self.dual_shape).to(self.device)
        
        for k in range(self.iterations):
            
            ### Dual map iteration step
            f_2 = f[:, 1:2, ...]
            # print('h',h.shape)
            # print('f_2',f_2.shape)
            # print('g',g.shape)
            h_new = self.dual_map[k](h, f_2, g, self.forward_operator)
            # print('h_new',h_new.shape)
            ### Primal map iteration step
            h = h_new[:, 0:1, ...]
            f = self.primal_map[k](h, f, self.adjoint_operator)
            h = h_new
            
            
        return f[:,0:1,:,:]
            


def summary_image_impl(writer, name, tensor, it):
    image = tensor[0, 0]
    image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))
    writer.add_image(name, image, it, dataformats='HW')
    
    
def summary_image(writer, name, tensor, it, window=False):
    summary_image_impl(writer, name + '/full', tensor, it)
    if window:
        summary_image_impl(writer, name + '/window', (tensor), it)   


def random_gf_pair(primal_space, forward_operator, mean, sigma):
    ellipsoids = [[1., np.random.rand(), np.random.rand(), 0.5*(np.random.rand()*2 - 1), 0.5*(np.random.rand()*2-1), 0.] for i in range(4)]
    f = odl.phantom.geometric.ellipsoid_phantom(primal_space, ellipsoids)
    f.asarray()[:] /= np.max(f.asarray())
    #g = K(f) + noise_level * odl.phantom.noise.white_noise(Y)
    g = forward_operator(f)
    g = g + np.random.normal(mean, sigma, size=(g.shape[0], g.shape[1]))
    f = torch.as_tensor(f)
    g = torch.as_tensor(g)
    return f, g

















