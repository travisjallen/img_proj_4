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
            new_coords = np.round(new_coords/new_coords[2])
            
            ## place the intensity at the old coords onto the location of the new coords
            location_rows = origin_r + new_coords[1]
            location_cols = origin_c + new_coords[0]
            canvas[int(location_rows),int(location_cols)] = image[i,j]

            ## record the location
            y_prime[i,j] = new_coords[0]
            x_prime[i,j] = new_coords[1]

    return canvas,y_prime,x_prime




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