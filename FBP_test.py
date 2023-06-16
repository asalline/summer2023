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
    angle_interval_points = odl.discr.grid.RectGrid(np.linspace(0, 2*np.pi, num_of_points)[:-1])
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

    # Sinogram of the image of ellipsoids
    sinogram = operator(image)

    # Adding some Gaussian noise to the sinogram to make the reconstruction harder
    noisy_sinogram = sinogram + np.random.normal(0, 0.01, size=(np.shape(sinogram)[0], np.shape(sinogram)[1]))

    return image, sinogram, noisy_sinogram


# Functions in the use
domain, codomain, operator, adjoint_operator = create_operator_and_space_FBP((64,64), 100)
image, sinogram, noisy_sinogram = random_ellipsoids_and_sinogram(domain, codomain, operator, adjoint_operator, 4)

# Reconstruction from the noisy image usin adjoint operator
reconstruction = adjoint_operator(noisy_sinogram)

# Plots
fig, ax = plt.subplots(2,2)
ax[0,0].imshow(image)
ax[0,1].imshow(sinogram)
ax[1,0].imshow(noisy_sinogram)
ax[1,1].imshow(reconstruction)

plt.show()


