## import necessary libraries
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

## import json reader
from read_json import read_json

corra,corrb,image_names,corrs = read_json("shelf_params.json")

plt.savefig('test_images/shelf_10_correspondences.png')
plt.show()


## 
## 