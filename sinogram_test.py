def create_operator_and_space():
    X = odl.uniform_discr([-1, -1], [1, 1], (64, 64), dtype='float32')
    
    ang_interval = odl.set.domain.IntervalProd(0, 2*np.pi)
    ang_interval_pts = odl.discr.grid.RectGrid(np.linspace(0, 2*np.pi, 64)[:-1])
    ang_partition = odl.discr.partition.RectPartition(ang_interval, ang_interval_pts)

    a = 1. * np.pi
    line_interval = odl.set.domain.IntervalProd(-a, a)
    line_interval_pts = odl.discr.grid.RectGrid(np.linspace(-a, a, 64)[:-1])
    line_partition = odl.discr.partition.RectPartition(line_interval, line_interval_pts)

    geom = odl.tomo.geometry.conebeam.FanBeamGeometry(ang_partition, line_partition, src_radius=2, det_radius=2)
    K = odl.tomo.RayTransform(X, geom, impl='astra_cuda')
    Y = K.range
    dK = K.adjoint
    return X, Y, K, dK


def random_gf_pair(X, Y, K, noise_level=0.1):
    ellipsoids = [[1., np.random.rand(), np.random.rand(), 0.5*(np.random.rand()*2 - 1), 0.5*(np.random.rand()*2-1), 0.] for i in range(4)]
    f = odl.phantom.geometric.ellipsoid_phantom(X, ellipsoids)
    #g = K(f) + noise_level * odl.phantom.noise.white_noise(Y)
    g = odl.phantom.poisson_noise(K(f) / noise_level) * noise_level
    return f, g


X, Y, K, dK = create_operator_and_space()
f, g = random_gf_pair(X, Y, K, noise_level=0.01)

f.show()
g.show()