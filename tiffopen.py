import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


bad_image = cv.imread('/home/asalline/Documents/summer2023/algorithms/Walnut1/Reconstructions/full_AGD_50_000062.tiff', cv.IMREAD_UNCHANGED)
good_image = cv.imread('/home/asalline/Documents/summer2023/algorithms/usable_walnuts/usable_full_AGD_50_000433.tiff', cv.IMREAD_UNCHANGED)

# print(np.max(good_image))

# bad_image = bad_image / np.max(bad_image)
# good_image = good_image / np.max(good_image)

print(f'dtype: {bad_image.dtype}, shape: {bad_image.shape}, min: {np.min(bad_image)}, max: {np.max(bad_image)}')
print(f'dtype: {good_image.dtype}, shape: {good_image.shape}, min: {np.min(good_image)}, max: {np.max(good_image)}')


nonzeros_bad = np.sum(np.sum(bad_image > 0.02))
print(nonzeros_bad, 501*501)
nonzeros_good = np.sum(np.sum(good_image > 0.02))
print(nonzeros_good, 501*501)


bad_norm = np.linalg.norm(bad_image, 1)
good_norm = np.linalg.norm(good_image, 1)

bad_histo = np.histogram(bad_image.ravel(), 501, [0,1])
good_histo = np.histogram(good_image.ravel(), 501, [0,1])

print(bad_norm, good_norm)

fig, ax = plt.subplots(2,2)

ax[0,0].imshow(bad_image)
ax[0,1].imshow(good_image)
# ax[1,0].hist(bad_histo)
# ax[1,1].hist(good_histo)
plt.show()

### USE 5000 NON-ZEROS AS A THRESHOLD


