### Post-processing image enhancement with Filtered Back Projection using UNet

###
### Needed packages: -odl
###                  -PyTorch
###                  -NumPy
###                  -matplotlib
###                  -UNet_functions.py (NEEDS ITS OWN PACKAGES EG. OpenCV)
###


### Importing packages and modules
import odl
import torch
import torch.nn as nn
import torch.optim as optim
from odl.contrib.torch import OperatorModule
#from odl.contrib import torch as odl_torch
#from torch.nn.utils import clip_grad_norm_
import numpy as np
from UNet_train_module import get_images, geometry_and_ray_trafo, UNet
from ellipsoids import *
import matplotlib.pyplot as plt

### Check if nvidia CUDA is available and using it, if not, the CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.cuda.empty_cache()


### Using functions from "UNet_train_module". Taking shape from images to produce
### odl parameters and getting ray transform operator and its adjoint.
shape = (100,100)
domain, geometry, ray_transform, output_shape = geometry_and_ray_trafo(setup='full', shape=shape, device=device, factor_lines = 2)
fbp_operator = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_transform, padding=1)

### Using odl functions to make odl operators into PyTorch modules
ray_transform_module = OperatorModule(ray_transform).to(device)
adjoint_operator_module = OperatorModule(ray_transform.adjoint).to(device)
fbp_operator_module = OperatorModule(fbp_operator).to(device)

### Here happens test images downloading. They are treated the same way as the training ones.
amount_of_images = 10
amount_of_ellipses = 10
test_images = torch.zeros((amount_of_images, ) + shape)
test_sinograms = torch.zeros((amount_of_images, ) + output_shape)
for k in range(amount_of_images):
    test_image, test_sinogram = random_ellipsoids_and_sinograms(domain, 1/amount_of_ellipses, amount_of_ellipses)
    test_images[k,:,:] = torch.as_tensor(test_image)
    test_sinograms[k,:,:] = torch.as_tensor(test_sinogram)

test_noisy_sinograms = torch.zeros((amount_of_images, ) + output_shape)
print(test_noisy_sinograms.shape)
mean = 0
percentage = 0.05

### Adding Gaussian noise to the sinograms
for k in range(np.shape(test_sinograms)[0]):
    sinogram_k = test_sinograms[k,:,:].cpu().detach().numpy()
    noise = np.random.normal(mean, sinogram_k.std(), sinogram_k.shape) * percentage
    noisy_sinogram = sinogram_k + noise
    test_noisy_sinograms[k,:,:] = torch.as_tensor(noisy_sinogram)

test_rec_images = torch.zeros((amount_of_images, ) + shape)
test_rec_images = fbp_operator_module(test_noisy_sinograms)

### Plotting one image from all and its sinogram and noisy sinogram
# image_number = 50
# noisy_sino = noisy_sinograms[image_number,0,:,:].cpu().detach().numpy()
# orig_sino = sinograms[image_number,0,:,:].cpu().detach().numpy()
# orig = rec_images[image_number,0,:,:].cpu().detach().numpy()

# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(orig)
# plt.subplot(1,3,2)
# plt.imshow(noisy_sino)
# plt.subplot(1,3,3)
# plt.imshow(orig_sino)
# plt.show()

### Setting UNet as model and passing it to the used device
UNet_network = UNet(in_channels=1, out_channels=1).to(device)

### Getting model parameters
UNet_parameters = list(UNet_network.parameters())

### Defining PSNR function.
def psnr(loss):
    
    psnr = 10 * np.log10(1.0 / loss+1e-10)
    
    return psnr

loss_train = nn.MSELoss()
loss_test = nn.MSELoss()

### Defining evaluation (test) function
def eval(net, g, f):

    test_loss = []
    
    ### Setting network into evaluation mode
    net.eval()
    test_loss.append(torch.sqrt(loss_test(net(g), f)).item())
    print(test_loss)
    out3 = net(g[0,None,:,:])

    return out3

### Setting up some lists used later
running_loss = []
running_test_loss = []

### Defining training scheme
def train_network(net, n_train=300, batch_size=25): #g_train, g_test, f_train, f_test, 

    ### Defining optimizer, ADAM is used here
    optimizer = optim.Adam(UNet_parameters, lr=0.001) #betas = (0.9, 0.99)
    
    ### Definign scheduler, can be used if wanted
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    
    images = torch.zeros((batch_size, ) + shape)
    sinograms = torch.zeros((batch_size, ) + output_shape)
    noisy_sinograms = torch.zeros((batch_size, ) + output_shape)
    rec_images = torch.zeros((batch_size, ) + shape)
    ### Here starts the itarting in training
    for i in range(n_train):
        
        ### Taking batch size amount of data pieces from the random 
        ### permutation of all training data

        for k in range(batch_size):
            # images[k,:,:], sinograms[k,:,:] = random_ellipsoids_and_sinograms(domain, 1/amount_of_ellipses, amount_of_ellipses)
            image, sinogram = random_ellipsoids_and_sinograms(domain, 1/amount_of_ellipses, amount_of_ellipses)
            images[k,:,:] = torch.as_tensor(image)
            sinograms[k,:,:] = torch.as_tensor(sinogram)


        for k in range(np.shape(sinograms)[0]):
            sinogram_k = sinograms[k,:,:].cpu().detach().numpy()
            noise = np.random.normal(mean, sinogram_k.std(), sinogram_k.shape) * percentage
            noisy_sinogram = sinogram_k + noise
            noisy_sinograms[k,:,:] = torch.as_tensor(noisy_sinogram)

        rec_images = torch.zeros((amount_of_images, ) + shape)
        rec_images = fbp_operator_module(noisy_sinograms)
        rec_images = rec_images[:,None,:,:]
        # print(rec_images.shape)
        
        net.train()

        ### Evaluating the network which not produces reconstructed images
        outs = net(rec_images)
        
        ### Setting gradient to zero
        optimizer.zero_grad()
        
        ### Calculating loss of the outputs
        loss = loss_train(outs, images)

        ### Calculating gradient
        loss.backward()
        
        ### Here gradient clipping could be used
        # torch.nn.utils.clip_grad_norm_(UNet_parameters, max_norm=1.0, norm_type=2)
        
        ### Taking optimizer step
        optimizer.step()
        # scheduler.step()

        ### Here starts the running tests
        if i % 100 == 0:
            ### Using predetermined test data to see how the outputs are
            ### in our neural network
            outs2 = net(test_rec_images[:,None,:,:])
            
            ### Calculating test loss with test data outputs
            test_loss = loss_test(outs2, test_images).item()
            train_loss = loss.item()
            running_loss.append(train_loss)
            running_test_loss.append(test_loss)
            
            ### Printing some data out
            if i % 1000 == 0:
                print(f'Iter {i}/{n_train} Train Loss: {train_loss:.2e}, Test Loss: {test_loss:.2e}, PSNR: {psnr(test_loss):.2f}') #, end='\r'
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(outs2[5,:,:].cpu().detach().numpy())
                plt.subplot(1,2,2)
                plt.imshow(test_images[5,:,:].cpu().detach().numpy())
                plt.show()

    ### After iterating taking one reconstructed image and its ground truth
    ### and showing them
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(outs[0,0,:,:].cpu().detach().numpy())
    plt.subplot(1,2,2)
    plt.imshow(images[0,0,:,:].cpu().detach().numpy())
    plt.show()

    ### Plotting running loss and running test loss
    plt.figure()
    plt.semilogy(running_loss)
    plt.semilogy(running_test_loss)
    plt.show()

    return running_loss, running_test_loss, net

### Calling training function to start the naural network training
running_loss, running_test_loss, net = train_network(UNet_network, n_train=1001, batch_size=1)

### Evaluating the network
net.eval()

### Taking images and plotting them to show how the neural network does succeed
image_number = int(np.random.randint(g_test.shape[0], size=1))
UNet_reconstruction = net(g_test[None,image_number,:,:,:])[0,0,:,:].cpu().detach().numpy()
ground_truth = f_test[image_number,0,:,:].cpu().detach().numpy()
noisy_reconstruction = g_test[image_number,0,:,:].cpu().detach().numpy()

plt.figure()
plt.subplot(1,3,1)
plt.imshow(noisy_reconstruction)
plt.subplot(1,3,2)
plt.imshow(UNet_reconstruction)
plt.subplot(1,3,3)
plt.imshow(ground_truth)
plt.show()

# torch.save(net.state_dict(), '/scratch2/antti/networks/'+'UNet1_005.pth')

