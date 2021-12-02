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
# print(len(base_identifier))

nconstituents = corrs[0][1][0].split('.')
nfirst_part = nconstituents[0]
# print(len(nfirst_part))

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

    ## now set up the linear system

    ## number of correspondences
    N = 0
    corra_shape = np.shape(corra)
    for j in range(corra_shape[0]):
        if (corra[j,0,i] != 0):
            ## then this is a correspondence
            N += 1
    
    ## initialize A matrix and b vector 
    A = np.zeros((int(2*N),8))
    b = np.zeros((int(2*N),1))

    ## fill A matrix
    A[0:N,0] = -corra[0:N,0,i]                  ## column 1, first half of rows
    A[0:N,1] = -corra[0:N,1,i]                  ## column 2, first half of rows
    A[0:N,2] = -1                               ## column 3, first half of rows
    A[0:N,3:5] = 0                              ## columns 4-6, first half of rows
    A[0:N,6] = corra[0:N,0,i]*corrb[0:N,0,i]    ## column 7, first half of rows
    A[0:N,7] = corra[0:N,1,i]*corrb[0:N,0,i]    ## column 8, first half of rows

    A[N:-1,0] = -corra[0:N,0,i]                 ## column 1, first half of rows
    A[N:-1,1] = -corra[0:N,1,i]                 ## column 2, first half of rows
    A[N:-1,2] = -1                              ## column 3, first half of rows
    A[N:-1,3:5] = 0                             ## columns 4-6, first half of rows
    A[N:-1,6] = corra[0:N,0,i]*corrb[0:N,0,i]   ## column 7, first half of rows
    A[N:-1,7] = corra[0:N,1,i]*corrb[0:N,0,i]   ## column 8, first half of rows

    print(A)