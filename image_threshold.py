import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import imageio

# Parameters
amount_of_pictures = 501
pixel_threshold = 0.02
threshold = 5000

for i in range(amount_of_pictures):
    while len(str(i)) < 3:
        i = str(0) + str(i)
    
    # Read the image
    image = cv.imread('/home/asalline/Documents/summer2023/algorithms/Walnut1/Reconstructions/full_AGD_50_000'+str(i)+'.tiff', cv.IMREAD_UNCHANGED)

    # Save the image
    if np.sum(np.sum(image > pixel_threshold)) > threshold:
        imageio.imwrite('/home/asalline/Documents/summer2023/algorithms/usable_walnuts/usable_full_AGD_50_000'+str(i)+'.tiff', image)