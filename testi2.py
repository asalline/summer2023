import torch
import torch.optim as optim
import torch.nn as nn
import odl
from odl.contrib.torch import OperatorModule
from odl.contrib import torch as odl_torch
import numpy as np
import matplotlib.pyplot as plt

# Defining the operators

# length = 100

# space = odl.uniform_discr([-1, -1], [1,1], [length, length], dtype='float32')

# forward_operator = odl.IdentityOperator(space)
# adjoint_operator = odl.IdentityOperator(space)

def create_operator_and_space(primal_shape, dual_shape):
        X = odl.uniform_discr([-1, -1], [1, 1], primal_shape, dtype='float32')
        
        ang_interval = odl.set.domain.IntervalProd(0, 2*np.pi)
        ang_interval_pts = odl.discr.grid.RectGrid(np.linspace(0, 2*np.pi, dual_shape[0]+1)[:-1])
        ang_partition = odl.discr.partition.RectPartition(ang_interval, ang_interval_pts)

        a = 1. * np.pi
        line_interval = odl.set.domain.IntervalProd(-a, a)
        line_interval_pts = odl.discr.grid.RectGrid(np.linspace(-a, a, dual_shape[1]))
        line_partition = odl.discr.partition.RectPartition(line_interval, line_interval_pts)

        geom = odl.tomo.geometry.conebeam.FanBeamGeometry(ang_partition, line_partition, src_radius=2, det_radius=2)
        K = odl.tomo.RayTransform(X, geom, impl='astra_cpu')
        Y = K.range
        dK = K.adjoint
        return X, Y, K, dK


def random_gf_pair(X, Y, K, noise_level=0.1):
    ellipsoids = [[1., np.random.rand(), np.random.rand(), 0.5*(np.random.rand()*2 - 1), 0.5*(np.random.rand()*2-1), 0.] for i in range(4)]
    f = odl.phantom.geometric.ellipsoid_phantom(X, ellipsoids)
    #g = K(f) + noise_level * odl.phantom.noise.white_noise(Y)
    g = odl.phantom.poisson_noise(K(f) / noise_level) * noise_level
    f = torch.as_tensor(f)
    g = torch.as_tensor(g)
    return f, g


primal_shape = (64,64)
dual_shape = (64,64)
primal_space, dual_space, forward_operator, adjoint_operator = create_operator_and_space(primal_shape, dual_shape)



# Generating data

amount_of_data = 210

g = torch.zeros((amount_of_data, ) + dual_shape)
x = torch.zeros((amount_of_data, ) + primal_shape)

for k in range(amount_of_data):

    x[k,:,:], g[k,:,:] = random_gf_pair(primal_space, dual_space, forward_operator, noise_level=0.01)





# f.show()
# g.show()



### CONSTANT BLOCK FUNCTION

# for k in range(amount_of_data):
#     x.append(np.zeros((length, length)))
#     x[k][int((length)/4):int((3*length)/4), int((length)/4):int((3*length)/4)] = 1

#     gaussian_noise = np.random.normal(0, 0.1, size=(length, length))
#     g.append(x[k] + gaussian_noise * np.max(np.max(x[k])))

# 0:int(length/4)  

x = torch.tensor(x)

g = torch.tensor(g)
g = g[:, None, :, :]
x = x[:, None, :, :]

g_train = g[0:199]
g_test = g[200:209]

x_train = x[0:199]
x_test = x[200:209]


# g = g[]:, :]
# print(np.shape(g))
# assert g.shape == (1,64,64)

# plt.figure(1)
# plt.imshow(g)
# cbar = plt.colorbar()
# plt.show()

class Dual_Network(nn.Module):
    def __init__(self, first_input, final_output):
        super().__init__()
        
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

    def forward(self, h, f, forward_operator, g):
        
        # print('dual')
        # print(np.shape(f))
        eval_f = forward_operator(f)
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
    def __init__(self, first_input, final_output):
        super().__init__()

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

    def forward(self, f, h_eval):

        # eval_h = adjoint_operator(h[None, :])
        x = torch.cat((f, h_eval), 1)
        x = f + self.primal_layers(x)

        return x
        
        

class Architecture(nn.Module):
    def __init__(self, forward_operator, adjoint_operator, primal, dual, primal_shape, dual_shape, iterations = 10):

        super(Architecture, self).__init__()

        # self.layers = nn.ModuleList()

        self.iterations = iterations
        self.forward_operator = forward_operator
        self.adjoint_operator = adjoint_operator
        self.primal = nn.ModuleList(primal)
        self.dual = nn.ModuleList(dual)
        self.primal_shape = primal_shape
        self.dual_shape = dual_shape


        self.green_arrow = OperatorModule(forward_operator)

        self.yellow_arrow = OperatorModule(adjoint_operator)

        # model = nn.Sequential(*[self.primal, self.dual])




    def forward(self, g, f_values = None, h_values = None):

        f_values = torch.zeros((g.shape[0], ) + self.primal_shape)
        h_values = torch.zeros((g.shape[0], ) + self.dual_shape)

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
            f_values = self.primal[k](f_values, self.yellow_arrow(h_1)) 

            h_values = h_new
    

        return f_values[:, 0:1, :, :]


iterations = 2
Dual_nets = [Dual_Network(7, 5) for i in range(iterations)]
Primal_nets = [Primal_Network(6, 5) for i in range(iterations)]

LPDnet = Architecture(forward_operator, adjoint_operator, Primal_nets, Dual_nets, (5, primal_shape[0], primal_shape[1]), (5, dual_shape[0], dual_shape[1]), iterations)

LPD_params = (LPDnet.parameters())


def loss_function(g, f_values):
    #loss = torch.mean(torch.linalg.norm(g - f_values)**2)
    loss = torch.mean((g - f_values)**2)
    return loss

# for k in range(amount_of_data):
#     test = LPDnet(g[k][None, :, :])

# print(test)

def eval(net, loss_function, g, x):

    test_loss = []

    net.eval()

    test_loss.append(torch.sqrt(loss_function(net(g), x[:, None, :, :][0:9])).item())

    print(test_loss)

    func = net(g[0,None,:,:])

    return func

    # for j, sample in enumerate(g):

    #     sample = sample[:, None, :, :]

    #     outs = net(sample)
    #     test_loss.append(loss_function(x[:,None,:,:][j], outs).item())
    #     print(f'Test Loss: {test_loss[j]**0.5}')


running_loss = []

def train_network(net, loss_function, x):
    n_train = 20


    optimizer = optim.Adam((LPDnet.parameters()), lr=0.001, betas = (0.9, 0.99))
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)

    net.train()

    for i in range(n_train):

        n_index = np.random.permutation(n_train)[:10]

        g_batch = g_train[n_index]
        x_batch = x[n_index]
        # print(np.shape(g_batch))
        outs = net(g_batch)
        # print('train outs')
        # print(np.shape(outs))

        # print(np.shape(x[:, None, :, :]))
        # print(np.shape(outs))
        loss = loss_function(x_batch, outs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        running_loss.append(loss.item())

        if i % 10 == 0:
            print(f'Loss: {running_loss[i]**0.5}')


    return running_loss, net





running_loss, test = train_network(LPDnet, loss_function, x_train)


test_data = g[99:109]
# print(np.shape(test_data))
# print(len(test_data))
func = eval(LPDnet, loss_function, g_test, x_test)

func = func.detach().numpy()
func = func[0,0,:,:]

orig = x_test[0, None, :, :]
orig = orig.detach().numpy()
orig = orig[0,0,:,:]

noisy = g[200, None, :, :]
noisy = noisy.detach().numpy()
noisy = noisy[0,0,:,:]

plt.figure()


plt.subplot(1,3,1)
plt.imshow(noisy)

plt.subplot(1,3,2)
plt.imshow(func)

plt.subplot(1,3,3)
plt.imshow(orig)

plt.show()

# plt.figure(1)

# plt.subplot(2,1,2)
# plt.imshow(test[0].detach().numpy())

# plt.subplot(2,2,2)
# plt.imshow(g[0])
# plt.show()

# plt.figure()

# plt.imshow(test[0].detach().numpy())
# plt.show()

# plt.figure()
# plt.imshow(g[0])
# plt.show()




