#######################
## Travis Allen
## CS 6640
## Project 4
#######################

## import necessary libraries
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from skimage import exposure

## import json reader
from read_json import read_json

## import module with user defined functions
import functions

## read the json file
corra,corrb,image_names,corrs = read_json("sign_params_4_mismatch.json")

## correct the image names
for i in range(len(image_names)):
    temp = image_names[i].split('.')
    image_names[i]=  temp[0]+'.png'

## read the first image
print(image_names[0])
img_0 = io.imread(image_names[0],as_gray=True)
img_0 = img_0/np.amax(img_0)

## determine its size
img_0_size = np.shape(img_0)
img_0_rows = img_0_size[0]
img_0_cols = img_0_size[1]
# img_0_layers = img_0_size[2]

## create a canvas
num_images = len(image_names)

## establish a factor of safety
fos = num_images + 5
canvas = np.zeros((fos*img_0_rows,fos*img_0_cols))
# canvas[:,:,:] = 0.5

## determine where to place the image
origin_r = int((fos*img_0_rows/2) - (img_0_rows/2))
origin_c = int((fos*img_0_cols/2) - (img_0_cols/2))
terminus_r = int(origin_r + img_0_rows)
terminus_c = int(origin_c + img_0_cols)

## place it
canvas[origin_r:terminus_r,origin_c:terminus_c] = img_0[:,:]

## identify the base image
base_image = image_names[0]
base_identifier = base_image.split('.')
base_identifier = base_identifier[0]

for i in range(num_images - 1):
    ## get the name of the second image
    img_name = corrs[i][1][0].split('.')
    img_identifier = img_name[0]
    
    ## store the second image and some information about it
    current_image = io.imread(img_identifier+'.png',as_gray=True)
    current_image_size = np.shape(current_image)
    
    ## match the histogram of the new image to the first image
    # current_image = exposure.match_histograms(current_image,img_0)

    # sometimes some small numbers get dragged around. Make them 0
    epsilon = 0.01
    for k in range(max(corra.shape)):
        if ((corra[k,0,i] + corra[k,1,i] < epsilon) and (corra[k,0,i] + corra[k,1,i] > -epsilon)):
            corra[k,:,i] = 0.0

    # sometimes some big numbers get dragged around. Make them 0
    sup = 1000000000
    for k in range(max(corra.shape)):
        if ((corra[k,0,i] + corra[k,1,i] > sup) or (corra[k,0,i] + corra[k,1,i] < -sup)):
            print("here")
            corra[k,:,i] = 0.0

    ## now set up the linear system

    ## number of correspondences
    N = 0
    for j in range(corra.shape[0]):
        if (corra[j,0,i] != 0):
            ## then this is a correspondence
            N += 1
    print(N)

    ## initialize A matrix and b vector 
    A = np.zeros((int(2*N),8))
    b = np.zeros((int(2*N),1))

    ## fill A matrix
    A[0:N,0] = -corra[0:N,0,i]                  ## column 1, first half of rows. -x
    A[0:N,1] = -corra[0:N,1,i]                  ## column 2, first half of rows. -y
    A[0:N,2] = -1                               ## column 3, first half of rows. -1
    A[0:N,3:5] = 0                              ## columns 4-6, first half of rows. 0's
    A[0:N,6] = corra[0:N,0,i]*corrb[0:N,0,i]    ## column 7, first half of rows. x*x'
    A[0:N,7] = corra[0:N,1,i]*corrb[0:N,0,i]    ## column 8, first half of rows. y*x'

    A[N:2*N,0] = 0                            ## columns 1-3, second half of rows. 0's
    A[N:2*N,1] = 0                            ## columns 1-3, second half of rows. 0's
    A[N:2*N,2] = 0                            ## columns 1-3, second half of rows. 0's
    A[N:2*N,3] = -corra[0:N,0,i]                ## column 4, second half of rows. -x
    A[N:2*N,4] = -corra[0:N,1,i]                ## column 5, second half of rows. -y
    A[N:2*N,5] = -1                           ## columns 6, second half of rows. -1
    A[N:2*N,6] = corra[0:N,0,i]*corrb[0:N,1,i]  ## column 7, second half of rows. x*y'
    A[N:2*N,7] = corra[0:N,1,i]*corrb[0:N,1,i]  ## column 8, second half of rows. y*y'
    
    ## fill b vector
    b[0:N,0] = -corrb[0:N,0,i]
    b[N:2*N,0] = -corrb[0:N,1,i]

    ## solve the system with SVD
    p = functions.svd_solve(A,b)
    
    ## make P matrix
    P = np.reshape(p,(-1,3))

    print(P)

    ## transform the image and place
    canvas,y_prime,x_prime = functions.transform(current_image,canvas,P,origin_r,origin_c)
    
    # new_canvas = functions.trim_canvas(canvas)

    # plt.figure()
    # plt.imshow(new_canvas,cmap='gray')
    # plt.axis('off')
    # plt.show()


new_canvas = functions.trim_canvas(canvas)

plt.figure(figsize=(12,7))
plt.imshow(new_canvas,cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.savefig('test_images/sign_4_mismatch.png')
plt.show()


