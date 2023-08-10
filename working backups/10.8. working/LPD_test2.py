
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
from LPD_test_functions2 import get_images, geometry_and_ray_trafo, LGD #, summary_image_impl, summary_image
import matplotlib.pyplot as plt
# import tensorboardX
# from torch.utils.tensorboard import SummaryWriter


### Check if nvidia CUDA is available and using it, if not, the CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.cuda.empty_cache()

### Using function "get_images" to import images from the path.
images = get_images('/scratch2/antti/summer2023/usable_walnuts', amount_of_images=100, scale_number=2)
### Converting images such that they can be used in calculations
images = np.array(images, dtype='float32')
images = torch.from_numpy(images).float().to(device)
# images = torch.from_numpy(images).float()

### Using functions from "UNet_functions". Taking shape from images to produce
### odl parameters and getting Radon transform operator and its adjoint.
shape = (np.shape(images)[1], np.shape(images)[2])
# shape = (64,64)
# print(shape)
domain, geometry, ray_transform, output_shape = geometry_and_ray_trafo(setup='full', shape=shape, device=device, factor_lines = 1)
adjoint_operator = ray_transform.adjoint
# print(domain)
# print(geometry)
# print(ray_transform)
# print(output_shape)
fbp_operator = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_transform, padding=1)
# print('domain',domain)
# print('ray', geometry)
### Using odl functions to make odl operators into PyTorch modules
ray_transform_module = OperatorModule(ray_transform).to(device)
adjoint_operator_module = OperatorModule(adjoint_operator).to(device)
fbp_operator_module = OperatorModule(fbp_operator).to(device)
#####################################
# n_train = 1500
# n_test = 10
# amount_of_data = n_train + n_test

# g = torch.zeros((amount_of_data, ) + output_shape)
# f = torch.zeros((amount_of_data, ) + shape)

# for k in range(amount_of_data):
#     if k % 100 == 0:
#         print(k)
#     f[k,:,:], g[k,:,:] = random_gf_pair(domain, ray_transform, mean=0.0005, sigma=0.0005**0.5)


# g = g[:, None, :, :].to(device)
# f = f[:, None, :, :].to(device)

# g_train = g[0:n_train]
# g_test = g[n_train:]

# f_train = f[:n_train]
# f_test = f[n_train:]

# hmm = adjoint_operator(g_train[5,0,:,:].cpu().detach().numpy())
# # hmm2 = ray_transform(f_train[5,0,:,:].cpu().detach().numpy())
# # hmm3 = fbp_operator(hmm2)

# plt.figure()
# plt.subplot(1,3,1)
# plt.imshow(g_train[5,0,:,:].cpu().detach().numpy())
# plt.subplot(1,3,2)
# plt.imshow(hmm)
# plt.subplot(1,3,3)
# # plt.imshow(hmm3)
# plt.imshow(f_train[5,0,:,:].cpu().detach().numpy())
# plt.show()
#############################
# sinograms = torch.zeros((images.shape[0], ) + output_shape)
# ### Making sinograms from the images using Radon transform module
# for k in range(images.shape[0]):
#     image = images[k,:,:].to(device)
#     image = image[None,None,:,:]
#     sinograms[k,:,:] = torch.as_tensor(ray_transform_module(image).cpu())
# sinograms = ray_transform(images)
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

mean = 0.000005
variance = 0.000001
sigma = variance ** 0.5

### Adding Gaussian noise to the sinograms. Here some problem solving is
### needed to make this smoother.
for k in range(np.shape(sinograms)[0]):
    #coeff = np.max(np.max(sinograms[k,:,:].cpu().detach().numpy()))
    # mean = 0.05 #* coeff
    # variance = 0.01 #* coeff
    # sigma = variance ** 0.5
    noisy_sinogram = sinograms[k,:,:].cpu().detach().numpy() + np.random.normal(mean, sigma, size=(sinograms.shape[1], sinograms.shape[2]))
    noisy_sinograms[k,:,:] = torch.as_tensor(noisy_sinogram)

### Using FBP to get reconstructed images from noisy sinograms
rec_images = fbp_operator_module(noisy_sinograms)

### All the data into same device
sinograms = sinograms[:,None,:,:].to(device)
noisy_sinograms = noisy_sinograms[:,None,:,:].to(device)
rec_images = rec_images[:,None,:,:].to(device)
images = images[:,None,:,:].to(device)

# sinograms = sinograms[:,None,:,:]
# noisy_sinograms = noisy_sinograms[:,None,:,:]
# rec_images = rec_images[:,None,:,:]
# images = images[:,None,:,:]



### Seperating the data into training and and testing data. 
### "g_" is data from reconstructed images and
### "f_" is data from ground truth images
# g_train = rec_images[0:n_train]
g_train = noisy_sinograms[0:n_train]
#g_test = rec_images[n_train:n_train+n_test]
f_train = images[0:n_train]
#f_test = images[n_train:n_train+n_test]
print('gtrain', g_train.shape)
print('ftrain', f_train.shape)


test_images = get_images('/scratch2/antti/summer2023/test_walnut', amount_of_images=10, scale_number=2)

test_images = np.array(test_images, dtype='float32')
test_images = torch.from_numpy(test_images).float().to(device)

test_sinograms = ray_transform_module(test_images)

test_noisy_sinograms = torch.zeros((test_sinograms.shape[0], ) + output_shape)
test_rec_images = torch.zeros((test_sinograms.shape[0], ) + shape)

for k in range(np.shape(test_sinograms)[0]):
    #coeff = np.max(np.max(sinograms[k,:,:].cpu().detach().numpy()))
    # mean = 0.005 #* coeff
    # variance = 0.001 #* coeff
    # sigma = variance ** 0.5
    test_noisy_sinogram = test_sinograms[k,:,:].cpu().detach().numpy() + np.random.normal(mean, sigma, size=(test_sinograms.shape[1], test_sinograms.shape[2]))
    test_noisy_sinograms[k,:,:] = torch.as_tensor(test_noisy_sinogram)
                                                  
    
test_rec_images = fbp_operator_module(test_noisy_sinograms)
    
test_sinograms = test_sinograms[:,None,:,:].to(device)
test_noisy_sinograms = test_noisy_sinograms[:,None,:,:].to(device)
test_rec_images = test_rec_images[:,None,:,:].to(device)
test_images = test_images[:,None,:,:].to(device)

# print(test_rec_images.shape)
# print(test_images.shape)

indices = np.random.permutation(test_rec_images.shape[0])[:5]
# g_test = test_rec_images[indices]
g_test = test_noisy_sinograms[indices]
f_test = test_images[indices]
# #######################

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
plt.colorbar()
plt.show()
###############################

primal_shape = (5, f_train.shape[2], f_train.shape[3])
dual_shape = (5, g_train.shape[2], g_train.shape[3])

### Setting UNet as model and passing it to the used device
LGD = LGD(adjoint_operator_module, ray_transform_module, primal_shape, dual_shape, in_channels_dual=3, out_channels_dual=1, \
          in_channels_primal=2, out_channels_primal=1, n_iter=10, device=device).to(device)

### Getting model parameters
LGD_parameters = list(LGD.parameters())

### Defining loss function. Can be changed if needed
def loss_function(g, f_values):
    
    loss = torch.mean((f_values - g)**2)
    return loss

loss_train = nn.MSELoss()
loss_test = nn.MSELoss()

### Defining evaluation (test) function
def eval(net, loss_function, f_images, g_sinograms, f_rec_images):

    test_loss = []
    
    ### Setting network into evaluation mode
    net.eval()
    test_loss.append(torch.sqrt(nn.MSELoss(net(f_rec_images, g_sinograms), f_images)).item())
    print(test_loss)
    out3 = net(f_rec_images[0,None,:,:], g_sinograms[0,None,:,:])

    return out3

### Setting up some lists used later
running_loss = []
running_test_loss = []

### Defining training scheme
def train_network(net, n_train=50000, batch_size=4):

    ### Defining optimizer, ADAM is used here
    optimizer = optim.Adam(LGD_parameters, lr=0.001, betas = (0.9, 0.99)) #betas = (0.9, 0.99)
    
    ### Definign scheduler, can be used if wanted
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_train)
    
    ### Setting network into training mode
    # net.train()

    ### Here starts the itarting in training
    for i in range(n_train):
        
        # scheduler.step()
        
        ### Taking batch size amount of data pieces from the random 
        ### permutation of all training data
        n_index = np.random.permutation(g_train.shape[0])[:batch_size]
        g_batch = g_train[n_index,:,:,:].to(device)
        f_batch = f_train[n_index].to(device)
        # f_batch2 = f_rec_images[n_index]
        
        net.train()
        
        optimizer.zero_grad()
        
        ### Taking out some enhanced images
        outs = net(f_batch, g_batch)
        
        # net.train()
        
        ### Setting gradient to zero
        # optimizer.zero_grad()
        
        ### Calculating loss of the outputs
        loss = loss_train(f_batch, outs)
        
        ### Calculating gradient
        loss.backward()
        
        ### Here gradient clipping could be used
        torch.nn.utils.clip_grad_norm_(LGD_parameters, max_norm=1.0, norm_type=2)
        
        ### Taking optimizer step
        optimizer.step()
        scheduler.step()

        ### Here starts the running tests
        if i % 100 == 0:
            
            ### Using predetermined test data to see how the outputs are
            ### in our neural network
            # net.eval()
            outs2 = net(f_test, g_test)
            # print('outs2', outs2.shape)
            ### Calculating test loss with test data outputs
            test_loss = loss_test(f_test, outs2).item()**0.5
            train_loss = loss.item() ** 0.5
            running_loss.append(train_loss)
            running_test_loss.append(test_loss)
            
            ### Printing some data out
            if i % 500 == 0:
                print(f'Iter {i}/{n_train} Train Loss: {train_loss:.2e}, Test Loss {test_loss:.2e}') #, end='\r'
                # print(f'Step lenght: {step_len[0]}') 
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(outs2[0,0,:,:].cpu().detach().numpy())
                plt.subplot(1,2,2)
                plt.imshow(f_test[0,0,:,:].cpu().detach().numpy())
                plt.show()
                
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
running_loss, running_test_loss, ready_to_eval = train_network(LGD, n_train=30000, \
                                                                  batch_size=1)

out3 = eval(ready_to_eval, loss_test, g_test, f_test)

### Taking images and plotting them to show how the neural network does succeed
image_number = int(np.random.randint(g_test.shape[0], size=1))
UNet_reconstruction = ready_to_eval(g_test[None,image_number,:,:,:])[0,0,:,:].cpu().detach().numpy()
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









