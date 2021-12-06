from numba import jit
import numpy as np

@jit(nopython=True)
def place_image(number_of_elements,canvas,primed_coords,origin_r,origin_c):
    for j in range(number_of_elements):
        ## place the image
        canvas[int(origin_r + primed_coords[j,0]),int(origin_c + primed_coords[j,1])] = primed_coords[j,2]
    return canvas

def svd_solve(A,b):
    ## find SVD with numpy's svd
    u,s,v_transpose = np.linalg.svd(A)

    ## the S given by np.linalg.svd is a list but we want a diagonal matrix
    singular_values = np.zeros(A.shape)
    for i in range(int(np.amin(A.shape))):
        singular_values[i,i] = 1/s[i]

    ## Compute the inverse of A (@ is matrix multiply)
    A_inv = v_transpose.T @ singular_values.T @ u.T

    ## now solve for P's
    P = A_inv @ b

    ## add the homogeneous coordinate to the vector of P's and return it
    P = np.append(P,1)
    return P

@jit(nopython=True)
def transform(number_of_elements,primed_coords,P):
    for j in range(number_of_elements):
        ## calculate starred coordinates from slide 39
        x_star = P[0,0]*primed_coords[j,1] + P[0,1]*primed_coords[j,0] + P[0,2]
        y_star = P[1,0]*primed_coords[j,1] + P[1,1]*primed_coords[j,0] + P[1,2]
        z_star = P[2,0]*primed_coords[j,1] + P[2,1]*primed_coords[j,0] + 1
        
        ## calculate primed coordinates
        primed_coords[j,1] = int(x_star/z_star)
        primed_coords[j,0] = int(y_star/z_star)
    
    return primed_coords

@jit(nopython=True)
def intensity_locations(current_image_size,primed_coords,current_image):
    count = 0
    for x in range(current_image_size[1]):
        for y in range(current_image_size[0]):
            ## store the x and y locations
            primed_coords[count,0] = y
            primed_coords[count,1] = x

            ## store intensity at that location (remember, (row, column))
            primed_coords[count,2] = current_image[y,x]

    return primed_coords

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