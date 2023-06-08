import torch
import torch.optim as optim
import torch.nn as nn
import odl
from odl.contrib.torch import OperatorModule
from odl.contrib import torch as odl_torch
import numpy as np
import matplotlib.pyplot as plt


def create_operator_and_space(primal_shape, dual_shape, device='cuda'):
    device = 'astra_' + device

    X = odl.uniform_discr([-1, -1], [1, 1], primal_shape, dtype='float32')
    
    ang_interval = odl.set.domain.IntervalProd(0, 2*np.pi)
    ang_interval_pts = odl.discr.grid.RectGrid(np.linspace(0, 2*np.pi, dual_shape[0]+1)[:-1])
    ang_partition = odl.discr.partition.RectPartition(ang_interval, ang_interval_pts)

    a = 1. * np.pi
    line_interval = odl.set.domain.IntervalProd(-a, a)
    line_interval_pts = odl.discr.grid.RectGrid(np.linspace(-a, a, dual_shape[1]))
    line_partition = odl.discr.partition.RectPartition(line_interval, line_interval_pts)

    geom = odl.tomo.geometry.conebeam.FanBeamGeometry(ang_partition, line_partition, src_radius=2, det_radius=2)
    K = odl.tomo.RayTransform(X, geom, impl=device)
    Y = K.range
    dK = K.adjoint
    return X, Y, K, dK


def random_gf_pair(X, Y, K, noise_level=0.1):
    ellipsoids = [[1., np.random.rand(), np.random.rand(), 0.5*(np.random.rand()*2 - 1), 0.5*(np.random.rand()*2-1), 0.] for i in range(4)]
    f = odl.phantom.geometric.ellipsoid_phantom(X, ellipsoids)
    f.asarray()[:] /= np.max(f.asarray())
    #g = K(f) + noise_level * odl.phantom.noise.white_noise(Y)
    g = odl.phantom.poisson_noise(K(f) / noise_level) * noise_level
    f = torch.as_tensor(f)
    g = torch.as_tensor(g)
    return f, g



class Dual_Network(nn.Module):
    def __init__(self, first_input, final_output, device='cuda'):
        super().__init__()
       
        self.device = device
        self.first_input = first_input
        self.final_output = final_output

        # Defining layers
        dual_layers = [
            nn.Conv2d(in_channels=self.first_input, out_channels=32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=self.final_output, kernel_size=3, padding=1)
        ]

        self.dual_layers = nn.Sequential(*dual_layers)
        self.to(device)

    def forward(self, h, f, forward_operator, g):
       
        # print('dual')
        # print(np.shape(f))
        eval_f = forward_operator(f).to(self.device)
        # print(np.shape(h))
        # print(np.shape(eval_f))
        # print(np.shape(g))
        # h = h[1,:,:]
        # h = h[None, :]
        # print(np.shape(h[None,0,:,:]))
        x = torch.cat((h, eval_f, g), 1)
        x = x.type(torch.float32)
        # print(np.shape(x))
        x = h + self.dual_layers(x)

        return x
       

class Primal_Network(nn.Module):
    def __init__(self, first_input, final_output, device='cuda'):
        super().__init__()

        self.device = device
        self.first_input = first_input
        self.final_output = final_output

        # Defining layers

        primal_layers = [
            nn.Conv2d(in_channels=self.first_input, out_channels=32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(in_channels=32, out_channels=self.final_output, kernel_size=3, padding=1)
        ]

        self.primal_layers = nn.Sequential(*primal_layers)
        self.to(device)

    def forward(self, f, h_eval):

        # eval_h = adjoint_operator(h[None, :])
        x = torch.cat((f, h_eval), 1)
        x = f + self.primal_layers(x)

        return x
       
       

class Architecture(nn.Module):
    def __init__(self, forward_operator, adjoint_operator, primal, dual, primal_shape, dual_shape, iterations = 10, device='cuda'):

        super(Architecture, self).__init__()

        # self.layers = nn.ModuleList()
        self.device = device
        self.iterations = iterations
        self.forward_operator = forward_operator
        self.adjoint_operator = adjoint_operator
        self.primal = nn.ModuleList(primal)
        self.dual = nn.ModuleList(dual)
        self.primal_shape = primal_shape
        self.dual_shape = dual_shape


        self.green_arrow = OperatorModule(forward_operator)

        self.yellow_arrow = OperatorModule(adjoint_operator)
        self.to(device)
        # model = nn.Sequential(*[self.primal, self.dual])




    def forward(self, g, f_values = None, h_values = None):

        f_values = torch.zeros((g.shape[0], ) + self.primal_shape).to(self.device)
        h_values = torch.zeros((g.shape[0], ) + self.dual_shape).to(self.device)
        

        # print('forw')
        # print(np.shape(h_values))
        # print(np.shape(f_values))
        # print(np.shape(g))
        for k in range(self.iterations):

           

            # First one applies the dual network
            f_2 = f_values[:, 1:2]
            h_new = self.dual[k](h_values, f_2, self.green_arrow, g)


            # Second one applies the primal network
            h_1 = h_values[:, 0:1]
            f_values = self.primal[k](f_values, self.yellow_arrow(h_1).to(self.device))

            h_values = h_new
   

        return f_values[:, 0:1, :, :]


