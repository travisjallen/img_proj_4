## import necessary libraries
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

## import json reader
from read_json import read_json

corra,corrb,image_names,corrs = read_json("sign_params_4.json")

plt.savefig('test_images/sign_4_correspondences.png')
plt.show()
