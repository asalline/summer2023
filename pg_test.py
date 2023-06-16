# """L1-regularized denoising using the proximal gradient solvers.

# Solves the optimization problem

#     min_x  lam || T(W.inverse(x)) - g ||_2^2 + lam * || x ||_1

# Where ``W`` is a wavelet operator, ``T`` is a parallel beam ray transform and
#  ``g`` is given noisy data.

# The proximal gradient solvers are also known as ISTA and FISTA.
# """

# import odl
# import numpy as np


# # --- Set up problem definition --- #


# # Reconstruction space: discretized functions on the rectangle
# # [-20, 20]^2 with 256 samples per dimension.
# space = odl.uniform_discr(
#     min_pt=[-20, -20], max_pt=[20, 20], shape=[256, 256], dtype='float32')

# # Make a parallel beam geometry with flat detector
# # Angles: uniformly spaced, n = 300, min = 0, max = pi
# angle_partition = odl.uniform_partition(0, np.pi, 300)
# # Detector: uniformly sampled, n = 300, min = -30, max = 30
# detector_partition = odl.uniform_partition(-30, 30, 300)
# geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# # Create the forward operator, and also the vectorial forward operator.
# ray_trafo = odl.tomo.RayTransform(space, geometry)


# # --- Generate artificial data --- #


# # Create phantom
# discr_phantom = odl.phantom.shepp_logan(space, modified=True)

# # Create sinogram of forward projected phantom with noise
# data = ray_trafo(discr_phantom)
# data += odl.phantom.white_noise(ray_trafo.range) * np.mean(data) * 0.1


# # --- Set up the inverse problem --- #


# # Create wavelet operator
# W = odl.trafos.WaveletTransform(space, wavelet='haar', nlevels=5)

# # The wavelets bases are normalized to constant norm regardless of scale.
# # since we want to penalize "small" wavelets more than "large" ones, we need
# # to weight by the scale of the wavelets.
# # The "area" of the wavelets scales as 2 ^ scale, but we use a slightly smaller
# # number in order to allow some high frequencies.
# scales = W.scales()
# Wtrafoinv = W.inverse * (1 / (np.power(1.7, scales)))

# # Create regularizer as l1 norm
# regularizer = 0.0005 * odl.solvers.L1Norm(W.range)

# # l2-squared norm of residual
# l2_norm_sq = odl.solvers.L2NormSquared(ray_trafo.range).translated(data)

# # Compose from the right with ray transform and wavelet transform
# data_discrepancy = l2_norm_sq * ray_trafo * Wtrafoinv

# # --- Select solver parameters and solve using proximal gradient --- #

# # Select step-size that gives convergence.
# gamma = 0.2

# # Optionally pass callback to the solver to display intermediate results
# # callback = (odl.solvers.CallbackPrintIteration() &
# #             odl.solvers.CallbackShow(step=5))


# # def callb(x):
# #     """Callback that displays the inverse wavelet transform of current iter."""
# #     callback(Wtrafoinv(x))


# # Run the algorithm (FISTA)
# x = data_discrepancy.domain.zero()
# odl.solvers.accelerated_proximal_gradient(
#     x, f=regularizer, g=data_discrepancy, niter=10, gamma=gamma)

# print(x)

# # Display images
# data.show(title='Data')
# x.show(title='Wavelet Coefficients')
# Wtrafoinv(x).show('Wavelet Regularized Reconstruction', force_show=True)


"""Total variation tomography using PDHG.

Solves the optimization problem

    min_x  1/2 ||A(x) - g||_2^2 + lam || |grad(x)| ||_1

Where ``A`` is a parallel beam forward projector, ``grad`` the spatial
gradient and ``g`` is given noisy data.

For further details and a description of the solution method used, see
https://odlgroup.github.io/odl/guide/pdhg_guide.html in the ODL documentation.
"""

import numpy as np
import odl

# --- Set up the forward operator (ray transform) --- #

# Reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300], dtype='float32')

# Make a parallel beam geometry with flat detector
# Angles: uniformly spaced, n = 360, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 360)
# Detector: uniformly sampled, n = 512, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 512)
geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

# Create the forward operator
ray_trafo = odl.tomo.RayTransform(reco_space, geometry)

# --- Generate artificial data --- #

# Create phantom
discr_phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create sinogram of forward projected phantom with noise
data = ray_trafo(discr_phantom)
data += odl.phantom.white_noise(ray_trafo.range) * np.mean(data) * 0.1

# --- Set up the inverse problem --- #

# Initialize gradient operator
gradient = odl.Gradient(reco_space)

# Column vector of two operators
op = odl.BroadcastOperator(ray_trafo, gradient)

# Do not use the f functional, set it to zero.
f = odl.solvers.ZeroFunctional(op.domain)

# Create functionals for the dual variable

# l2-squared data matching
l2_norm = odl.solvers.L2NormSquared(ray_trafo.range).translated(data)

# Isotropic TV-regularization i.e. the l1-norm
l1_norm = 0.015 * odl.solvers.L1Norm(gradient.range)

# Combine functionals, order must correspond to the operator K
g = odl.solvers.SeparableSum(l2_norm, l1_norm)

# --- Select solver parameters and solve using PDHG --- #

# Estimated operator norm, add 10 percent to ensure ||K||_2^2 * sigma * tau < 1
op_norm = 1.1 * odl.power_method_opnorm(op)

niter = 10  # Number of iterations
tau = 1.0 / op_norm  # Step size for the primal variable
sigma = 1.0 / op_norm  # Step size for the dual variable

# Optionally pass callback to the solver to display intermediate results
# callback = (odl.solvers.CallbackPrintIteration(step=10) &
#             odl.solvers.CallbackShow(step=10))

# Choose a starting point
x = op.domain.zero()

# Run the algorithm
odl.solvers.pdhg(x, f, g, op, niter=niter, tau=tau, sigma=sigma)

# Display images
discr_phantom.show(title='Phantom')
data.show(title='Simulated Data (Sinogram)')
x.show(title='TV Reconstruction', force_show=True)



