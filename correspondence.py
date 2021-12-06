import numpy as np
import skimage as sk
from skimage import io
import matplotlib.pyplot as plt

## read the images
shelf_0 = io.imread('test_images/shelf_0.png',as_gray=True)
shelf_1 = io.imread('test_images/shelf_1.png',as_gray=True)

plt.figure()
plt.imshow(shelf_0,cmap='gray')
plt.show()
plt.figure()
plt.imshow(shelf_1,cmap='gray')
plt.show()