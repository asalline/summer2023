import odl
import numpy as np
import matplotlib.pyplot as plt

def geometry_and_ray_trafo(setup='full', min_domain_corner=[-1,-1], max_domain_corner=[1,1], \
                           shape=(100,100), source_radius=2, detector_radius=1, \
                           dtype='float32', device='cpu', factor_lines = 1):

    device = 'astra_' + device
    print(device)
    domain = odl.uniform_discr(min_domain_corner, max_domain_corner, shape, dtype=dtype)

    if setup == 'full':
        angles = odl.uniform_partition(0, 2*np.pi, 360)
        lines = odl.uniform_partition(-1*np.pi, np.pi, int(1028/factor_lines))
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (360, int(1028/factor_lines))
    elif setup == 'sparse':
        angle_measurements = 100
        line_measurements = int(512/factor_lines)
        angles = odl.uniform_partition(0, 2*np.pi, angle_measurements)
        lines = odl.uniform_partition(-1*np.pi, np.pi, line_measurements)
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (angle_measurements, line_measurements)
    elif setup == 'limited':
        starting_angle = 0
        final_angle = np.pi * 3/4
        angles = odl.uniform_partition(starting_angle, final_angle, 360)
        lines = odl.uniform_partition(-1*np.pi, np.pi, int(512/factor_lines))
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (int(360), int(512/factor_lines))
        
    # domain = odl.uniform_discr(min_domain_corner, max_domain_corner, output_shape, dtype=dtype)

    ray_transform = odl.tomo.RayTransform(domain, geometry, impl=device)

    return domain, geometry, ray_transform, output_shape


### Function that generates random or user-determined ellipsoids in an image and a sinogram of that image.
###
### Inputs: domain: The domain/space one gets from ODL. Domain has to correspond to
###                 reference rectangle [-1,-1]x[1,1]. Also [-n,-n]x[n,n] rectangles
###                 are valid since there is a scaling in centering the ellipsoids
###         intensity: Intensity of the ellipsoid
###         amount_of_ellipses: How many ellipses one wants to generate, default=10
###         width: Width of the ellipsoid, default=random between 0 and 1
###         height: height of the ellipsoid, default=random between 0 and 1
###         x_center: center of the ellipsoid w.r.t. x-axis, default=random between
###                   minimum and maximum value of the domain rectangle.
###         y_center: center of the ellipsoid w.r.t. y-axis, default=random between
###                   minimum and maximum value of the domain rectangle.
###         rotation: amount of rotation in ellipsoid, must be given in radians, default=
###                   random between 0 and 2*pi
### Outputs: ellipsoids_image: 2D-array which now contains the ellipsoids, can be plotted
###          to see how it looks
###          sinogram: sinogram of the ellipsoids_image
###
### It is god to notice that if one sets width, height, x_center, y_center and rotation
### by themself, if one or more parameters needs to be randomized, it should be done by
### for example letting width = np.random.rand(amount_of_ellipsoids).
### Also every array must be exactly amount_of_ellipsoids long.
def random_ellipsoids_and_sinograms(domain, intensity, amount_of_ellipses=10, width='rand', \
                                     height='rand', x_center='rand', y_center='rand', rotation='rand'):

    if width=='rand' and height=='rand' and x_center=='rand' and y_center=='rand' and rotation=='rand':
        a = np.min(domain.min_pt)
        b = np.max(domain.max_pt)
        x_center = (np.random.rand(amount_of_ellipses) * (b - a) + a) / b
        y_center = (np.random.rand(amount_of_ellipses) * (b - a) + a) / b
        ellipsoids = [[intensity, np.random.rand(), np.random.rand(), x_center[i], y_center[i], \
                       np.random.rand() * 2*np.pi] for i in range(amount_of_ellipses)]
        ellipsoids_image = odl.phantom.geometric.ellipsoid_phantom(domain, ellipsoids)
        sinogram = ray_transform(ellipsoids_image)
        ellipsoids_image = ellipsoids_image / np.max(np.max(ellipsoids_image))
        sinogram = sinogram / np.max(np.max(sinogram))

    else:
        ellipsoids = [[intensity, width[i], height[i], x_center[i], y_center[i], rotation[i]] for i in range(amount_of_ellipses)]
        ellipsoids_image = odl.phantom.geometric.ellipsoid_phantom(domain, ellipsoids)
        sinogram = ray_transform(ellipsoids_image)
        ellipsoids_image = ellipsoids_image / np.max(np.max(ellipsoids_image))
        sinogram = sinogram / np.max(np.max(sinogram))

    return ellipsoids_image, sinogram

setup = 'full'
min_domain_corner = [-1,-1]
max_domain_corner = [1,1]
shape = (100,100)
source_radius = 2
detector_radius = 1
dtype = 'float32'
device = 'cpu'
factor_lines = 2


domain, geometry, ray_transform, output_shape = geometry_and_ray_trafo(setup, min_domain_corner, max_domain_corner, \
                                                                       shape, source_radius, detector_radius, dtype, \
                                                                         device, factor_lines)

# width = [0.1,0.2,0.3,0.4]
# height = [0.4,0.3,0.2,0.1]
# x_center = [0,0.1,0.2,0.3]/np.max(domain.max_pt)
# y_center = [0,0.3,0.2,0.1]/np.max(domain.max_pt)
# rotation = [0,0,0,0]

amount_of_ellipses = 10
# amount_of_ellipses = len(width)

ell, sino = random_ellipsoids_and_sinograms(domain, 1/amount_of_ellipses, amount_of_ellipses) #, width, height, x_center, y_center, rotation
# ell.show()
# sino.show()
# plt.show()

