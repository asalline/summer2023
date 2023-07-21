### Importing packages and modules
import odl
import torch
#import torch.nn as nn
import torch.optim as optim
from odl.contrib.torch import OperatorModule
#from odl.contrib import torch as odl_torch
#from torch.nn.utils import clip_grad_norm_
import numpy as np
from new_test_functions import geometry_and_ray_trafo, UNet, random_ellipsoids_and_sinogram
import matplotlib.pyplot as plt


### Check if nvidia CUDA is available and using it, if not, the CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.cuda.empty_cache()

# ### Using function "get_images" to import images from the path.
# images = get_images('/scratch2/antti/summer2023/usable_walnuts', amount_of_images=2000, scale_number=4)
# ### Converting images such that they can be used in calculations
# images = np.array(images, dtype='float32')
# images = torch.from_numpy(images).float().to(device)


### Using functions from "UNet_functions". Taking shape from images to produce
### odl parameters and getting Radon transform operator and its adjoint.
# shape = (np.shape(images)[1], np.shape(images)[2])
shape = (64,64)
domain, geometry, ray_transform, output_shape = geometry_and_ray_trafo(setup='full', shape=shape, device=device, factor_lines = 4)
fbp_operator = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_transform, padding=1)


### Using odl functions to make odl operators into PyTorch modules
ray_transform_module = OperatorModule(ray_transform).to(device)
adjoint_operator_module = OperatorModule(ray_transform.adjoint).to(device)
fbp_operator_module = OperatorModule(fbp_operator).to(device)

n_train = 1000
n_test = int(np.floor(n_train*0.05))
amount_of_data = n_train + n_test

images = torch.zeros((amount_of_data, ) + shape)
noisy_sinograms = torch.zeros((amount_of_data, ) + (100,100))
rec_images = torch.zeros((amount_of_data, ) + shape)

for k in range(amount_of_data):
    image = random_ellipsoids_and_sinogram(domain, ray_transform, 4)
    image = torch.as_tensor(image)
    images[k,:,:] = image
    if k % 50 == 0:
        print(f'{k}/{amount_of_data}')


sinograms = ray_transform_module(images)

for j in range(amount_of_data):
    # sinogram = ray_transform(image)
    mean = 0.00
    variance = 0.001
    sigma = variance ** 0.5
    noisy_sinogram = sinograms[j,:,:] + np.random.normal(mean, sigma, size=(np.shape(sinograms[j,:,:])[0], np.shape(sinograms[j,:,:])[1]))
    # rec_image = fbp_operator(noisy_sinogram)
    # rec_image = torch.as_tensor(rec_image)
    # rec_images[k,:,:] = rec_image
    noisy_sinogram = torch.as_tensor(noisy_sinogram)
    noisy_sinograms[k,:,:] = noisy_sinogram
    if k % 50 == 0:
        print(f'{j}/{amount_of_data}')

rec_images = fbp_operator_module(noisy_sinograms)
# print(noisy_sinograms[2].shape)
noisy_sinograms = noisy_sinograms[:,None,:,:].to(device)
rec_images = rec_images[:,None,:,:].to(device)
images = images[:,None,:,:].to(device)



### Seperating the data into training and and testing data. 
### "g_" is data from reconstructed images and
### "f_" is data from ground truth images
g_train = rec_images[0:n_train]
#g_test = rec_images[n_train:n_train+n_test]
f_train = images[0:n_train]
#f_test = images[n_train:n_train+n_test]
g_test = rec_images[n_train:]
f_test = images[n_train:]


### Setting UNet as model and passing it to the used device
UNet = UNet(in_channels=1, out_channels=1).to(device)

### Getting model parameters
UNet_parameters = list(UNet.parameters())

### Defining loss function. Can be changed if needed
def loss_function(g, f_values):
    
    loss = torch.mean((g - f_values)**2)
    return loss

### Defining evaluation (test) function
def eval(net, loss_function, g, f):

    test_loss = []
    
    ### Setting network into evaluation mode
    net.eval()
    test_loss.append(torch.sqrt(loss_function(net(g), f)).item())
    print(test_loss)
    out3 = net(g[0,None,:,:])

    return out3

### Setting up some lists used later
running_loss = []
running_test_loss = []

### Defining training scheme
def train_network(net, loss_function, g_train, g_test, f_train, f_test, n_train=300, batch_size=25):

    ### Defining optimizer, ADAM is used here
    optimizer = optim.Adam(UNet_parameters, lr=0.001) #betas = (0.9, 0.99)
    
    ### Definign scheduler, can be used if wanted
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    
    ### Setting network into training mode
    

    ### Here starts the itarting in training
    for i in range(n_train):
        
        ### Taking batch size amount of data pieces from the random 
        ### permutation of all training data
        n_index = np.random.permutation(g_train.shape[0])[:batch_size]
        g_batch = g_train[n_index,:,:,:]
        f_batch = f_train[n_index]
        
        ### Taking out some enhanced images
        net.train()
        
        optimizer.zero_grad()
        
        outs = net(g_batch)
        
        ### Setting gradient to zero
        # optimizer.zero_grad()
        
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
            outs2 = net(g_test)
            
            ### Calculating test loss with test data outputs
            test_loss = loss_function(f_test, outs2).item()**0.5
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
running_loss, running_test_loss, ready_to_eval = train_network(UNet, loss_function, g_train, g_test, f_train, f_test, n_train=25000, batch_size=4)

### Evaluating the network
out3 = eval(ready_to_eval, loss_function, g_test, f_test)

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