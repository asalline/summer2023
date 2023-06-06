from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import odl
from odl.contrib.torch import OperatorAsModule
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch import optim
import torchvision
from torchvision import datasets, transforms
torch.manual_seed(123);  # reproducibility

# Load training and test data (from the official MNIST example,
# https://github.com/pytorch/examples/blob/master/mnist/main.py)
trafo = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

load = True
if load == True:
    dset_train = datasets.MNIST('./data', train=True, download=True, transform=trafo)
    train_loader = torch.utils.data.DataLoader(dset_train, batch_size=50, shuffle=True)

    dset_test = datasets.MNIST('./data', train=False, transform=trafo)
    test_loader = torch.utils.data.DataLoader(dset_test, batch_size=50, shuffle=True)



### Create ODL objects

space = odl.uniform_discr([-14, -14], [14, 14], [28, 28], dtype='float32')
geometry = odl.tomo.parallel_beam_geometry(space, num_angles=5)
fwd_op = odl.tomo.RayTransform(space, geometry)
fbp_op = odl.tomo.fbp_op(fwd_op)



### Make those objects modules

fwd_op_mod = OperatorAsModule(fwd_op)
fwd_op_adj_mod = OperatorAsModule(fwd_op.adjoint)
fbp_op_mod = OperatorAsModule(fbp_op)



### Some helper functions



def generate_data(images):
    """Create noisy projection data from images.
    
    The data is generated according to ::
        
        data = fwd_op(images) + noise
        
    where ``noise`` is standard white noise.
    
    Parameters
    ----------
    images : `Variable`, shape ``(B, C, 28, 28)``
        Input images for the data generation.
        
    Returns
    -------
    data : `Variable`, shape ``(B, C, 5, 41)``
        Projection data stack.
    """
    torch.manual_seed(123)
    data = fwd_op_mod(images)
    data += Variable(torch.randn(data.shape)).type_as(data)
    return data


def show_image_matrix(image_batches, titles=None, indices=None, **kwargs):
    """Visualize a 2D set of images arranged in a grid.

    This function shows a 2D grid of images, where the i-th column
    shows images from the i-th batch. The typical use case is to compare
    results of different approaches with the same data, or to compare
    against a ground truth.

    Parameters
    ----------
    image_batches : sequence of `Tensor` or `Variable`
        List containing batches of images that should be displayed.
        Each tensor should have the same shape after squeezing, except
        for the batch axis.
    titles : sequence of str, optional
        Titles for the colums in the plot. By default, titles are empty.
    indices : sequence of int, optional
        Object to select the subset of the images that should be shown.
        The subsets are determined by slicing along the batch axis, i.e.,
        as ``displayed = image_batch[indices]``. The default is to show
        everything.
    kwargs :
        Further keyword arguments that are passed on to the Matplotlib
        ``imshow`` function.
    """
    import matplotlib.pyplot as plt

    if indices is None:
        displayed_batches = image_batches
    else:
        displayed_batches = [batch[indices] for batch in image_batches]

    displayed_batches = [batch.data if isinstance(batch, Variable) else batch
                         for batch in displayed_batches]

    nrows = len(displayed_batches[0])
    ncols = len(displayed_batches)

    if titles is None:
        titles = [''] * ncols

    figsize = 2
    fig, rows = plt.subplots(
        nrows, ncols, sharex=True, sharey=True,
        figsize=(ncols * figsize, figsize * nrows))

    if nrows == 1:
        rows = [rows]

    for i, row in enumerate(rows):
        if ncols == 1:
            row = [row]
        for name, batch, ax in zip(titles, displayed_batches, row):
            if i == 0:
                ax.set_title(name)
            ax.imshow(batch[i].squeeze(), **kwargs)
            ax.set_axis_off()
    plt.show()



### Generate test data



# Get a batch of test images and generate test projection data
for i, (images, _) in enumerate(test_loader):
    if i == 1:
        break

test_images = Variable(images)
test_data = generate_data(test_images)



### FBP reconstruction

fbp_recos = fbp_op_mod(test_data)
print('Average error:', F.mse_loss(fbp_recos, test_images).data[0] / len(test_images))

# Display examples
results = [test_images, fbp_recos]
titles = ['Truth', 'FBP']
show_image_matrix(results, titles, indices=slice(10, 20), clim=[0, 1], cmap='bone')



