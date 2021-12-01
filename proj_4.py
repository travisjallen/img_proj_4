#######################
## Travis Allen
## CS 6640
## Project 4
#######################

## import necessary libraries
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

## import json reader
from read_json import read_json

## read the json file
corra,corrb,image_names,corrs = read_json("mosaic_params.json")

## correct the image names
for i in range(len(image_names)):
    temp = image_names[i].split('.')
    image_names[i]=  temp[0]+'.png'

## read the first image
img_0 = io.imread(image_names[0])
img_0 = img_0/np.amax(img_0)

## determine its size
img_0_size = np.shape(img_0)
img_0_rows = img_0_size[0]
img_0_cols = img_0_size[1]
img_0_layers = img_0_size[2]

## create a canvas
num_images = len(image_names)

## establish a factor of safety
fos = num_images + 1
canvas = np.ndarray((fos*img_0_rows,fos*img_0_cols,img_0_layers))
canvas[:,:,:] = 0.5

## determine where to place the image
origin_r = int((fos*img_0_rows/2) - (img_0_rows/2))
origin_c = int((fos*img_0_cols/2) - (img_0_cols/2))
terminus_r = int(origin_r + img_0_rows)
terminus_c = int(origin_c + img_0_cols)

## place it
canvas[origin_r:terminus_r,origin_c:terminus_c,:] = img_0[:,:,:]

## plot it
# plt.figure()
# plt.imshow(canvas,cmap='gray')
# plt.axis('off')
# plt.show()

base_image = image_names[0]
base_identifier = base_image.split('.')
base_identifier = base_identifier[0]
print(len(base_identifier))

nconstituents = corrs[0][1][0].split('.')
nfirst_part = nconstituents[0]
print(len(nfirst_part))

for i in range(num_images - 1):
    ## determine if the relative order of the correlations
    img_name = corrs[i][1][0].split('.')
    img_identifier = img_name[0]
    
    ## check to see if the image has 3 characters before the dot
    if (len(img_identifier) == 3):
        ## if it does, pull the last one off
        img_identifier = img_identifier.rstrip(img_identifier[-1])

    ## now check to see if the base image is the second image in the correlation
    if (base_identifier == img_identifier):
        ## if so, swap the columns of the correlation
        temp = corra[:,:,i]
        corra[:,:,i] = corrb[:,:,i]
        corrb[:,:,i] = temp