from numba import jit
import numpy as np
from scipy import interpolate



def svd_solve(A,b):
    ## find SVD with numpy's svd
    u,s,v_transpose = np.linalg.svd(A)

    ## the S given by np.linalg.svd is a list but we want a diagonal matrix
    singular_values = np.zeros(A.shape)
    for i in range(int(np.amin(A.shape))):
        singular_values[i,i] = 1/s[i]

    ## Compute the inverse of A (@ is matrix multiply)
    A_inv = v_transpose.transpose() @ singular_values.transpose() @ u.transpose()

    ## now solve for P's
    P = A_inv @ b

    ## add the homogeneous coordinate to the vector of P's and return it
    P = np.append(P,1)
    return P



def transform(image,canvas,P,origin_r,origin_c):
    '''Takes image, canvas, P matrix, location of image origin, and places perspectively equivalent transformed image on canvas.'''
    ## make new coordinate arrays
    x_prime = np.zeros(image.shape)
    y_prime = np.zeros(image.shape)

    ## loop through the original image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            ## record the original coordinates in a homogeneous coordinate
            original_coords = np.array([[j,i,1]]).transpose()

            ## matrix multiply original coordinates by P
            new_coords = P @ original_coords

            ## Now divide by last coordinate for perspective equivalence. This will produce floats, so round then typecast to int when placing
            new_coords = np.ceil(new_coords/new_coords[2])
            
            ## place the intensity at the old coords onto the location of the new coords
            location_rows = origin_r + new_coords[1]
            location_cols = origin_c + new_coords[0]
            canvas[int(location_rows),int(location_cols)] = image[i,j]

            ## record the location
            y_prime[i,j] = new_coords[0]
            x_prime[i,j] = new_coords[1]

    return canvas,y_prime,x_prime



def mosaic_interp(P,image,x_prime,y_prime):
    ## determine x and y coords of original image
    x_coords = np.zeros((image.shape[1],1))
    y_coords = np.zeros((image.shape[0],1))
    for u in range(image.shape[1]):
        x_coords[u,0] = u

    for v in range(image.shape[0]):
            y_coords[v,0] = v

    ## calculate the interpolating function
    f_interp = interpolate.interp2d(x_coords,y_coords,image)

    ## find p inverse
    P_inv = np.linalg.inv(P)

    ## make vectors to store float coords
    x_float = np.zeros(x_coords.shape)
    y_float = np.zeros(x_coords.shape)

    for c in range(x_coords.shape[0]):
        for d  in range(y_coords.shape[1]):
            ## make the current coordinate a homogeneous coordinate
            current_coordinate = np.array([[x_prime[c,d],y_prime[c,d],1]]).transpose()

            ## un-transform the homogeneous coordinate
            un_transformed_coordinate = P_inv @ current_coordinate

            

def in_polygon(x_prime,y_prime):
    ## find region corners
    corners = np.zeros((2,4))
    corners[0,0] = x_prime[0,0]     ## upper left x
    corners[0,1] = y_prime[0,0]     ## upper left y

    corners[1,0] = x_prime[0,-1]    ## upper right x
    corners[1,1] = y_prime[0,-1]    ## upper left y
    
    corners[2,0] = x_prime[-1,0]    ## lower left x
    corners[2,1] = y_prime[-1,0]    ## upper left y
    
    corners[3,0] = x_prime[-1,-1]   ## lower right x
    corners[3,1] = y_prime[-1,-1]   ## upper left y



@jit(nopython=True)
def feather(canvas,x_prime,y_prime,origin_c,origin_r):
    ## box filter the left edge
    for idx in range(x_prime.shape[0]):
        ## find the locations
        loc_r = int(origin_r + x_prime[idx,0])
        loc_c = int(origin_c + y_prime[idx,0])
        
        ## compute some partial sums for a box filter
        ps = canvas[loc_r-2,loc_c-2] + canvas[loc_r-2,loc_c-1] + canvas[loc_r-2,loc_c] + canvas[loc_r-2,loc_c+1] + canvas[loc_r-2,loc_c+2]
        ps = canvas[loc_r-1,loc_c-2] + canvas[loc_r-1,loc_c-1] + canvas[loc_r-1,loc_c] + canvas[loc_r-1,loc_c+1] + canvas[loc_r-2,loc_c+2] + ps
        ps = canvas[loc_r,loc_c-2] + canvas[loc_r,loc_c-1] + canvas[loc_r,loc_c] + canvas[loc_r,loc_c+1] + canvas[loc_r,loc_c+2] + ps
        ps = canvas[loc_r+1,loc_c-2] + canvas[loc_r+1,loc_c-1] + canvas[loc_r+1,loc_c] + canvas[loc_r+1,loc_c+1] + canvas[loc_r+1,loc_c+2]+ ps
        ps = canvas[loc_r+2,loc_c-2] + canvas[loc_r+2,loc_c-1] + canvas[loc_r+2,loc_c] + canvas[loc_r+2,loc_c+1] + canvas[loc_r+2,loc_c+2]+ ps
        
        ## average the result
        avg = ps/25
        
        ## print it
        canvas[loc_r,loc_c] = avg


    ## box filter the right edge
    for idx in range(x_prime.shape[0]):
        ## find the locations
        loc_r = int(origin_r + x_prime[idx,-1])
        loc_c = int(origin_c + y_prime[idx,-1])
        
        ## compute some partial sums for a box filter
        ps = canvas[loc_r-2,loc_c-2] + canvas[loc_r-2,loc_c-1] + canvas[loc_r-2,loc_c] + canvas[loc_r-2,loc_c+1] + canvas[loc_r-2,loc_c+2]
        ps = canvas[loc_r-1,loc_c-2] + canvas[loc_r-1,loc_c-1] + canvas[loc_r-1,loc_c] + canvas[loc_r-1,loc_c+1] + canvas[loc_r-2,loc_c+2] + ps
        ps = canvas[loc_r,loc_c-2] + canvas[loc_r,loc_c-1] + canvas[loc_r,loc_c] + canvas[loc_r,loc_c+1] + canvas[loc_r,loc_c+2] + ps
        ps = canvas[loc_r+1,loc_c-2] + canvas[loc_r+1,loc_c-1] + canvas[loc_r+1,loc_c] + canvas[loc_r+1,loc_c+1] + canvas[loc_r+1,loc_c+2]+ ps
        ps = canvas[loc_r+2,loc_c-2] + canvas[loc_r+2,loc_c-1] + canvas[loc_r+2,loc_c] + canvas[loc_r+2,loc_c+1] + canvas[loc_r+2,loc_c+2]+ ps
        
        ## average the result
        avg = ps/25
        
        ## print it
        canvas[loc_r,loc_c] = avg
    
    ## box filter the top edge
    for idx in range(x_prime.shape[1]):
        ## find the locations
        loc_r = int(origin_r + x_prime[0,idx])
        loc_c = int(origin_c + y_prime[0,idx])
        
        ## compute some partial sums for a box filter
        ps = canvas[loc_r-2,loc_c-2] + canvas[loc_r-2,loc_c-1] + canvas[loc_r-2,loc_c] + canvas[loc_r-2,loc_c+1] + canvas[loc_r-2,loc_c+2]
        ps = canvas[loc_r-1,loc_c-2] + canvas[loc_r-1,loc_c-1] + canvas[loc_r-1,loc_c] + canvas[loc_r-1,loc_c+1] + canvas[loc_r-2,loc_c+2] + ps
        ps = canvas[loc_r,loc_c-2] + canvas[loc_r,loc_c-1] + canvas[loc_r,loc_c] + canvas[loc_r,loc_c+1] + canvas[loc_r,loc_c+2] + ps
        ps = canvas[loc_r+1,loc_c-2] + canvas[loc_r+1,loc_c-1] + canvas[loc_r+1,loc_c] + canvas[loc_r+1,loc_c+1] + canvas[loc_r+1,loc_c+2]+ ps
        ps = canvas[loc_r+2,loc_c-2] + canvas[loc_r+2,loc_c-1] + canvas[loc_r+2,loc_c] + canvas[loc_r+2,loc_c+1] + canvas[loc_r+2,loc_c+2]+ ps
        
        ## average the result
        avg = ps/25
        
        ## print it
        canvas[loc_r,loc_c] = avg


    ## box filter the bottom edge
    for idx in range(x_prime.shape[1]):
        ## find the locations
        loc_r = int(origin_r + x_prime[-1,idx])
        loc_c = int(origin_c + y_prime[-1,idx])
        
        ## compute some partial sums for a box filter
        ps = canvas[loc_r-2,loc_c-2] + canvas[loc_r-2,loc_c-1] + canvas[loc_r-2,loc_c] + canvas[loc_r-2,loc_c+1] + canvas[loc_r-2,loc_c+2]
        ps = canvas[loc_r-1,loc_c-2] + canvas[loc_r-1,loc_c-1] + canvas[loc_r-1,loc_c] + canvas[loc_r-1,loc_c+1] + canvas[loc_r-2,loc_c+2] + ps
        ps = canvas[loc_r,loc_c-2] + canvas[loc_r,loc_c-1] + canvas[loc_r,loc_c] + canvas[loc_r,loc_c+1] + canvas[loc_r,loc_c+2] + ps
        ps = canvas[loc_r+1,loc_c-2] + canvas[loc_r+1,loc_c-1] + canvas[loc_r+1,loc_c] + canvas[loc_r+1,loc_c+1] + canvas[loc_r+1,loc_c+2]+ ps
        ps = canvas[loc_r+2,loc_c-2] + canvas[loc_r+2,loc_c-1] + canvas[loc_r+2,loc_c] + canvas[loc_r+2,loc_c+1] + canvas[loc_r+2,loc_c+2]+ ps
        
        ## average the result
        avg = ps/25
        
        ## print it
        canvas[loc_r,loc_c] = avg

    return canvas



def trim_canvas(canvas):
    ## determine the size of the original canvas
    image_size = np.shape(canvas)
    
    ## find first row with nonzero element
    for i in range(image_size[0]):
        ## sum the elements in the row i
        if (np.sum(canvas[i,:]) != 0):
            ## then there is at least one nonzero pixel
            first_nonzero_row = i

    ## find the first column with nonzero element
    for i in range(image_size[1]):
        ## sum the elements in the row i
        if (np.sum(canvas[:,i]) != 0):
            ## then there is at least one nonzero pixel
            first_nonzero_col = i

    ## find last nonzero row
    for i in range(image_size[0]-1, 0, -1):
        ## sum the elements in the row i
        if (np.sum(canvas[i,:]) != 0):
            ## then there is at least one nonzero pixel
            last_nonzero_row = i

    ## find last nonzero column
    for i in range(image_size[1]-1, 0, -1):
        ## sum the elements in the row i
        if (np.sum(canvas[:,i]) != 0):
            ## then there is at least one nonzero pixel
            last_nonzero_col = i

    ## determine the new canvas size
    new_canvas_rows = first_nonzero_row - last_nonzero_row
    new_canvas_cols = first_nonzero_col - last_nonzero_col
    
    ## build the new canvas
    new_canvas = np.zeros((new_canvas_rows,new_canvas_cols))

    ## place the old ROI on the new canvas
    new_canvas[:,:] = canvas[last_nonzero_row:first_nonzero_row, last_nonzero_col:first_nonzero_col]

    return new_canvas