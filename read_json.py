import json
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def read_json(filename):
    with open(filename) as f:
        data = json.load(f)

    # print(f"The images are {data['Input files']}")
    corrs = data["Correspondences"]
    # print(f"There are {len(corrs)} sets of correspondences")

    ## make arrays to store corra and corrb
    num_corrs = np.zeros((1,len(corrs)))
    for i in range(len(corrs)):
        num_corrs[0,i] = len(corrs[i][0][1])
    
    corra = np.ndarray((int(np.amax(num_corrs)),2,len(corrs)))
    corrb = np.ndarray((int(np.amax(num_corrs)),2,len(corrs)))


    for i in range(len(corrs)):
        print(f"There are {len(corrs[i][0][1])} correspdondences between image {corrs[i][0][0]} and image {corrs[i][1][0]}")
        corra[0:int(num_corrs[0,i]),:,i] = np.asarray(corrs[i][1][1])
        corrb[0:int(num_corrs[0,i]),:,i] = np.asarray(corrs[i][0][1])   
        # print(corra[:,:,i])
        # print(corrb.T)
        imageNamea = corrs[i][0][0]
        imageNameb = corrs[i][1][0]
        imageNamea = imageNamea.split('.')
        imageNameb = imageNameb.split('.')
        imagea = io.imread(imageNamea[0]+'.png')
        imageb = io.imread(imageNameb[0]+'.png')
        fig, ax = plt.subplots(1,2)
        fig.set_size_inches(10.5,6.5)
        ax[0].imshow(imagea)
        ax[1].scatter(corra[:,0,i],corra[:,1,i],c='#ff7f0e')
        ax[1].imshow(imageb)
        ax[0].scatter(corrb[:,0,i],corrb[:,1,i],c='#ff7f0e')
        plt.savefig(str(i)+'.png')
    #     print(corra.shape)
    #     print(corrb.shape)
    # print(f"The output file is {data['Output file']}")

    ## return the correspondences and the image names
    image_names = data['Input files']
    output = data['Output file']

    return corra,corrb,image_names,corrs,output
