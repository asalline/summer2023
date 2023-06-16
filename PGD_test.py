import odl
import numpy as np
import torch
import matplotlib.pyplot as plt


# Function that creates needed ODL variables
def create_operator_and_space_FBP(shape, num_of_points, device='cpu'):

    device = 'astra_' + device

    # Defining discrete uniform domain
    domain = odl.uniform_discr([-1,-1], [1,1], shape, dtype='float32')

    # Defining some angle things for ray transform
    angle_interval = odl.set.domain.IntervalProd(0, 2*np.pi)
    angle_interval_points = odl.discr.grid.RectGrid(np.linspace(0, 2*np.pi, num_of_points)) #[:-1]
    angles = odl.discr.partition.RectPartition(angle_interval, angle_interval_points)

    # Defining some line things for ray transform
    line_interval = odl.set.domain.IntervalProd(-1*np.pi, 1*np.pi)
    line_interval_points = odl.discr.grid.RectGrid(np.linspace(-1*np.pi, 1*np.pi, num_of_points))
    lines = odl.discr.partition.RectPartition(line_interval, line_interval_points)

    # Defining fan beam geometry to be used in ray transform
    geometry = odl.tomo.geometry.conebeam.FanBeamGeometry(angles, lines, src_radius=2, det_radius=2)

    # Ray transform operator
    operator = odl.tomo.RayTransform(domain, geometry, impl=device)

    # Defining codomain just in case
    codomain = operator.range

    # And lastly the adjoint operator
    adjoint_operator = odl.tomo.analytic.filtered_back_projection.fbp_op(operator, padding=1)

    return domain, codomain, operator, adjoint_operator


def random_ellipsoids_and_sinogram(domain, codomain, operator, adjoint_operator, num_of_ellipsoids):

    # Create user-wanted-amount of random ellipsoids within the domain
    ellipsoids = [[1., np.random.rand(), np.random.rand(), 0.5*(np.random.rand()*2 - 1), 0.5*(np.random.rand()*2 - 1), 0] for i in range(num_of_ellipsoids)]

    # Ellipsoids as matrix, which can be plotted as an image
    image = odl.phantom.geometric.ellipsoid_phantom(domain, ellipsoids)
    image = torch.as_tensor(image)
    # Sinogram of the image of ellipsoids
    sinogram = operator(image)

    # Adding some Gaussian noise to the sinogram to make the reconstruction harder
    noisy_sinogram = sinogram + np.random.normal(0, 0.01, size=(np.shape(sinogram)[0], np.shape(sinogram)[1]))

    return image, sinogram, noisy_sinogram


# Functions in the use
domain, codomain, operator, adjoint_operator = create_operator_and_space_FBP((64,64), 64)
image, sinogram, noisy_sinogram = random_ellipsoids_and_sinogram(domain, codomain, operator, adjoint_operator, 4)

# plt.figure()
# plt.imshow(noisy_sinogram)
# plt.show()

# domain = torch.as_tensor(domain)
# noisy_sinogram = torch.as_tensor(noisy_sinogram)

grad = odl.Gradient(domain)
# print('grad', grad.shape)
l2_norm = odl.solvers.L2NormSquared(operator.range).translated(noisy_sinogram)
print(np.shape(noisy_sinogram))
# data_discrepancy = l2_norm.translated(noisy_sinogram)
data_discrepancy = l2_norm * operator

# regularizer = 0.05 * odl.solvers.GroupL1Norm(grad.range, 2) * grad
# regularizer = 0.05 * odl.solvers.L1Norm(grad.range) *grad
regularizer = 0.05 * odl.solvers.L2NormSquared(noisy_sinogram)

x = data_discrepancy.domain.zero()

print('x', np.shape(x))
print('regularizer', regularizer)
print('data', data_discrepancy)

print('##################')
print(data_discrepancy.domain)
print(regularizer.domain)
# print(x.domain)


reconstruction = odl.solvers.nonsmooth.proximal_gradient_solvers.proximal_gradient(x, f = regularizer, g = data_discrepancy, niter = 200, gamma=0.001)

### "no proximal operator implemented for funcitonal", Jevgenija tells in here -> https://github.com/odlgroup/odl/issues/1610 <- that 
###                                                    'the proximal is not implemented for the composition of l1 and gradient'

### F JA G OVAT ERI DOMAINISSA!!!

# print(reconstruction)

reconstruction.show

# reconstruction.cpu().detach().numpy()
# print(np.dtype(reconstruction))

# plt.figure()
# plt.imshow(reconstruction)
# plt.show()


### https://github.com/odlgroup/odl/blob/master/examples/solvers/proximal_gradient_denoising.py

