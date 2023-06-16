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
    adjoint_operator = odl.tomo.RayTransform.adjoint

    return domain, codomain, operator, adjoint_operator


def random_ellipsoids_and_sinogram(domain, codomain, operator, adjoint_operator, num_of_ellipsoids):

    # Create user-wanted-amount of random ellipsoids within the domain
    ellipsoids = [[1., np.random.rand(), np.random.rand(), 0.5*(np.random.rand()*2 - 1), 0.5*(np.random.rand()*2 - 1), 0] for i in range(num_of_ellipsoids)]

    # Ellipsoids as matrix, which can be plotted as an image
    image = odl.phantom.geometric.ellipsoid_phantom(domain, ellipsoids)
    # image = torch.as_tensor(image)
    # Sinogram of the image of ellipsoids
    sinogram = operator(image)

    # Adding some Gaussian noise to the sinogram to make the reconstruction harder
    noisy_sinogram = sinogram + np.random.normal(0, 0.07, size=(np.shape(sinogram)[0], np.shape(sinogram)[1]))

    return image, sinogram, noisy_sinogram

domain, codomain, operator, adjoint_operator = create_operator_and_space_FBP((100,100), 100)
image, sinogram, noisy_sinogram = random_ellipsoids_and_sinogram(domain, codomain, operator, adjoint_operator, 4)

# plt.figure()
# plt.imshow(image)
# plt.show()

# Setting up the inverse problem

identity_mapping = odl.IdentityOperator(domain)
gradient = odl.Gradient(domain)

L = odl.BroadcastOperator(operator, gradient)

l2_norm_squared = odl.solvers.L2NormSquared(operator.range).translated(noisy_sinogram)
l1_norm = 0.005 * odl.solvers.L1Norm(gradient.range)
g = odl.solvers.SeparableSum(l2_norm_squared, l1_norm)

# Non-negativity constraint
# f = odl.solvers.IndicatorNonnegativity(domain)
f = odl.solvers.ZeroFunctional(L.domain)

# Selecting optimization parameters tau and lambda
operator_norm = 1.1 * odl.power_method_opnorm(L, xstart = noisy_sinogram, maxiter = 4)
tau = 1.0 / operator_norm
sigma = tau
niter = 500

x = domain.zero()

odl.solvers.pdhg(x, f, g, L, tau = tau, sigma = sigma, niter = niter)

# print(sinogram)

# adjoint_operator = odl.tomo.RayTransform.inverse
# adjoint_operator(x)

# print(type(x))
sinogram.show('Sinogram without noise')
noisy_sinogram.show('Noisy sinogram')
image.show('Ground truth')
x.show('Reconstruction with {} iterations'.format(niter) , force_show=True)
# adjoint_operator(x).show('asd3')
plt.show()



# plt.figure()
# plt.imshow(x)
# plt.show()


