import odl
import odl.discr as discr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# python -m pip install numpy==1.23.5
# 

x = Image.open('mario.jpg').convert('L')

# plt.figure(2)
# plt.imshow(x)

# plt.show()


x_np = np.array(x)

X = odl.uniform_discr([-1,-1], [1,1], x_np.shape, dtype='float32')

x = X.element(x_np)

# x.show()

# None

# plt.show()

Id = odl.IdentityOperator(X)
gradX = discr.Gradient(X)


gx = gradX(x)


gx.show()
plt.show()

None


gradX_adj = gradX.adjoint
gradX_adj(gx).show()

plt.show()





