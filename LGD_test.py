### Post-processing image enhancement with Filtered Back Projection using UNet
###
### Author: Antti Sällinen
### Date: July 2023
###
### This script is designed to take in an images (preferrably CT-scans),
### using radon transform to get the sinograms of those images, adding noise
### to those sinograms and then with filtered back projection (FBP)
### reconstructing those images. FBP does not perform very well under some
### noise so deep learning approach is suitable. Known architecture UNet is
### used to train neural network to enhance the noisy image which FBP produces.
### UNet that is used is thre layers deep.
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
#import torch.nn as nn
import torch.optim as optim
from odl.contrib.torch import OperatorModule
#from odl.contrib import torch as odl_torch
#from torch.nn.utils import clip_grad_norm_
import numpy as np
from LGD_test_functions import get_images, geometry_and_ray_trafo, LGD
import matplotlib.pyplot as plt

### Check if nvidia CUDA is available and using it, if not, the CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

### Using function "get_images" to import images from the path.
images = get_images('/scratch2/antti/summer2023/usable_walnuts', amount_of_images='all', scale_number=4)
### Converting images such that they can be used in calculations
images = np.array(images, dtype='float32')
images = torch.from_numpy(images).float().to(device)

### Using functions from "UNet_functions". Taking shape from images to produce
### odl parameters and getting Radon transform operator and its adjoint.
shape = (np.shape(images)[1], np.shape(images)[2])
domain, geometry, ray_transform, output_shape = geometry_and_ray_trafo(setup='full', shape=shape, device=device, factor_lines = 4)

fbp_operator = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_transform, padding=1)

partial_x = odl.PartialDerivative(domain, axis=0)
partial_y = odl.PartialDerivative(domain, axis=1)
regularizer = OperatorModule(partial_x.adjoint * partial_x + partial_y.adjoint * partial_y).to(device)


### Using odl functions to make odl operators into PyTorch modules
ray_transform_module = OperatorModule(ray_transform).to(device)
adjoint_operator_module = OperatorModule(ray_transform.adjoint).to(device)
fbp_operator_module = OperatorModule(fbp_operator).to(device)

### Making sinograms from the images using Radon transform module
sinograms = ray_transform_module(images)

### Allocating used tensors
noisy_sinograms = torch.zeros((sinograms.shape[0], ) + output_shape)
rec_images = torch.zeros((sinograms.shape[0], ) + shape)

### Defining variables which define the amount of training and testing data
### being used. The training_scale is between 0 and 1 and controls how much
### training data is taken from whole data
training_scale = 0.95
amount_of_data = sinograms.shape[0]
n_train = int(np.floor(training_scale * amount_of_data))
n_test = int(np.floor(amount_of_data - n_train))

### Adding Gaussian noise to the sinograms. Here some problem solving is
### needed to make this smoother.
for k in range(np.shape(sinograms)[0]):
    #coeff = np.max(np.max(sinograms[k,:,:].cpu().detach().numpy()))
    mean = 0.005 #* coeff
    variance = 0.001 #* coeff
    sigma = variance ** 0.5
    noisy_sinogram = sinograms[k,:,:].cpu().detach().numpy() + np.random.normal(mean, sigma, size=(sinograms.shape[1], sinograms.shape[2]))
    noisy_sinograms[k,:,:] = torch.as_tensor(noisy_sinogram)

### Using FBP to get reconstructed images from noisy sinograms
rec_images = fbp_operator_module(noisy_sinograms)

### All the data into same device
sinograms = sinograms[:,None,:,:].to(device)
noisy_sinograms = noisy_sinograms[:,None,:,:].to(device)
rec_images = rec_images[:,None,:,:].to(device)
images = images[:,None,:,:].to(device)

### Seperating the data into training and and testing data. 
### "g_" is data from reconstructed images and
### "f_" is data from ground truth images

f_images = images[0:n_train]
print('f_images', f_images.shape)
g_sinograms = noisy_sinograms[0:n_train]
f_rec_images = rec_images[0:n_train]

test_images = get_images('/scratch2/antti/summer2023/test_walnut', amount_of_images=n_test, scale_number=4)

test_images = np.array(test_images, dtype='float32')
test_images = torch.from_numpy(test_images).float().to(device)

test_sinograms = ray_transform_module(test_images)

test_noisy_sinograms = torch.zeros((test_sinograms.shape[0], ) + output_shape)
test_rec_images = torch.zeros((test_sinograms.shape[0], ) + shape)

for k in range(np.shape(test_sinograms)[0]):
    #coeff = np.max(np.max(sinograms[k,:,:].cpu().detach().numpy()))
    mean = 0.005 #* coeff
    variance = 0.001 #* coeff
    sigma = variance ** 0.5
    test_noisy_sinogram = test_sinograms[k,:,:].cpu().detach().numpy() + np.random.normal(mean, sigma, size=(test_sinograms.shape[1], test_sinograms.shape[2]))
    test_noisy_sinograms[k,:,:] = torch.as_tensor(test_noisy_sinogram)
                                                  
    
test_rec_images = fbp_operator_module(test_noisy_sinograms)
    
test_sinograms = test_sinograms[:,None,:,:].to(device)
test_noisy_sinograms = test_noisy_sinograms[:,None,:,:].to(device)
test_rec_images = test_rec_images[:,None,:,:].to(device)
test_images = test_images[:,None,:,:].to(device)

print(test_rec_images.shape)
print(test_images.shape)

indices = np.random.permutation(test_rec_images.shape[0])[:10]
f_test_images = test_images[indices]
g_test_sinograms = test_noisy_sinograms[indices]
f_test_rec_images = test_rec_images[indices]

### Plotting one image from all and its sinogram and noisy sinogram
image_number = 50
noisy_sino = noisy_sinograms[image_number,0,:,:].cpu().detach().numpy()
orig_sino = sinograms[image_number,0,:,:].cpu().detach().numpy()
orig = rec_images[image_number,0,:,:].cpu().detach().numpy()

plt.figure()
plt.subplot(1,3,1)
plt.imshow(orig)
plt.subplot(1,3,2)
plt.imshow(noisy_sino)
plt.subplot(1,3,3)
plt.imshow(orig_sino)
plt.show()

image_number = 25
noisy_sino = test_noisy_sinograms[image_number,0,:,:].cpu().detach().numpy()
orig_sino = test_sinograms[image_number,0,:,:].cpu().detach().numpy()
orig = test_rec_images[image_number,0,:,:].cpu().detach().numpy()

plt.figure()
plt.subplot(1,3,1)
plt.imshow(orig)
plt.subplot(1,3,2)
plt.imshow(noisy_sino)
plt.subplot(1,3,3)
plt.imshow(orig_sino)
plt.show()

memory = 0
### Setting UNet as model and passing it to the used device
LGD = LGD(adjoint_operator_module, ray_transform_module, regularizer, f_images, \
          g_sinograms, f_rec_images, in_channels=2, out_channels=1, memory=memory, n_iter=5).to(device)

### Getting model parameters
LGD_parameters = list(LGD.parameters())

### Defining loss function. Can be changed if needed
def loss_function(g, f_values):
    
    loss = torch.mean((f_values - g)**2)
    return loss

### Defining evaluation (test) function
def eval(net, loss_function, f_images, g_sinograms, f_rec_images):

    test_loss = []
    
    ### Setting network into evaluation mode
    net.eval()
    test_loss.append(torch.sqrt(loss_function(net(f_rec_images, g_sinograms), f_images)).item())
    print(test_loss)
    out3 = net(f_rec_images[0,None,:,:], g_sinograms[0,None,:,:])

    return out3

### Setting up some lists used later
running_loss = []
running_test_loss = []

### Defining training scheme
def train_network(net, loss_function, f_images, g_sinograms, f_rec_images, f_test_rec_images, \
                  f_test_images, g_test_sinograms, n_train=50000, batch_size=4, memory=0):

    ### Defining optimizer, ADAM is used here
    optimizer = optim.Adam(LGD_parameters, lr=0.001) #betas = (0.9, 0.99)
    
    ### Definign scheduler, can be used if wanted
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    
    ### Setting network into training mode
    net.train()

    ### Here starts the itarting in training
    for i in range(n_train):
      
        ### Taking batch size amount of data pieces from the random 
        ### permutation of all training data
        n_index = np.random.permutation(g_sinograms.shape[0])[:batch_size]
        g_batch = g_sinograms[n_index,:,:,:]
        f_batch = f_images[n_index]
        f_batch2 = f_rec_images[n_index]
        
        ### Taking out some enhanced images
        outs = net(f_batch2, g_batch)
        
        ### Setting gradient to zero
        optimizer.zero_grad()
        
        ### Calculating loss of the outputs
        loss = loss_function(f_batch, outs)

        # loss = torch.from_numpy(loss)

        #loss.requires_grad=True
        
        ### Calculating gradient
        loss.backward()
        
        ### Here gradient clipping could be used
        # torch.nn.utils.clip_grad_norm_(UNet_parameters, max_norm=1.0, norm_type=2)
        
        ### Taking optimizer step
        optimizer.step()
        # scheduler.step()
        
        #running_loss.append(loss.item())

        ### Here starts the running tests
        if i % 100 == 0:
          
            ### Using predetermined test data to see how the outputs are
            ### in our neural network
            outs2 = net(f_test_rec_images, g_test_sinograms)
            # print('outs2', outs2.shape)
            ### Calculating test loss with test data outputs
            test_loss = loss_function(f_test_images, outs2).item()**0.5
            train_loss = loss.item() ** 0.5
            running_loss.append(train_loss)
            running_test_loss.append(test_loss)
            
            ### Printing some data out
            if i % 1000 == 0:
                print(f'Iter {i}/{n_train} Train Loss: {train_loss:.2e}, Test Loss {test_loss:.2e}') #, end='\r'

    ### After iterating taking one reconstructed image and its ground truth
    ### and showing them
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(outs[0,0,:,:].cpu().detach().numpy())
    plt.subplot(1,2,2)
    plt.imshow(f_batch[0,0,:,:].cpu().detach().numpy())
    plt.show()

    ### Plotting running loss and running test loss
    plt.figure()
    plt.semilogy(running_loss)
    plt.semilogy(running_test_loss)
    plt.show()

    return running_loss, running_test_loss, net

### Calling training function to start the naural network training
running_loss, running_test_loss, ready_to_eval = train_network(LGD, loss_function, f_images, \
                                                                  g_sinograms, f_rec_images, \
                                                                  f_test_rec_images, f_test_images, \
                                                                  g_test_sinograms, n_train=50000, \
                                                                  batch_size=1, memory=memory)

### Evaluating the network
out3 = eval(ready_to_eval, loss_function, f_test_images, g_test_sinograms, f_test_rec_images)

### Taking images and plotting them to show how the neural network does succeed
image_number = int(np.random.randint(g_test_sinograms.shape[0], size=1))
LGD_reconstruction = ready_to_eval(f_test_rec_images[None,image_number,:,:,:], g_test_sinograms[None,image_number,:,:,:])[0,0,:,:].cpu().detach().numpy()
ground_truth = f_test_images[image_number,0,:,:].cpu().detach().numpy()
noisy_reconstruction = f_test_rec_images[image_number,0,:,:].cpu().detach().numpy()

plt.figure()
plt.subplot(1,3,1)
plt.imshow(noisy_reconstruction)
plt.subplot(1,3,2)
plt.imshow(LGD_reconstruction)
plt.subplot(1,3,3)
plt.imshow(ground_truth)
plt.show()


