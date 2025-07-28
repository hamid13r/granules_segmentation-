import cc3d
import numpy as np
import mrcfile
import glob
from scipy.spatial import cKDTree
import skimage.filters as filters
import os
import matplotlib.pyplot as plt
import skimage.segmentation as segment
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.optimize as opt
from copy import copy
import pandas as pd
from matplotlib.widgets import Slider


     

def monogaussian(x, h, c, w):
    return h*np.exp(-(x-c)**2/(2*w**2))
def dual_gaussian(x, h1, c1, w1, h2, c2, w2): # p = h1, c1, w1, h2, c2, w2
    return monogaussian(x,h1,c1,w1)+monogaussian(x,h2,c2,w2)


input_dir = "scaled/"
mask_dir = "masks/"
output_dir= "segmentation_output/"
structuring_element = ball(3)

#make sure the output directory exists

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
apix = 21.1
write_output = 0
# Define a 3D structuring element (ball-shaped)
#structuring_element = ball(2)
tomo_i = 0
filename_list = glob.glob(input_dir + "/*.mrc")
threshold_df = pd.DataFrame()
while tomo_i < len(filename_list):
    filename = filename_list[tomo_i]
    basename = os.path.basename(filename)
    basename = basename[:basename.find(".mrc")]
    maskname = glob.glob(mask_dir + "/" + basename + "*")[0] # Get the first mask file that matches the basename
    print(tomo_i, filename)
    tomo_i = tomo_i + 1
    print(f"Hi! Processing file: {basename}")
    

#    with mrcfile.open(f'{filename}','r') as mrc_in:
    mrc_in = mrcfile.open(f'{filename}', mode='r')
    #mask_in = mrcfile.open(f'{maskname[0]}', mode='r')
#print(mrc_in.voxel_size)
    tomo_data = mrc_in.data
    tomo_data = tomo_data.astype(np.float32)
    print(f"number of nans: {np.sum(np.isnan(tomo_data))}")
    print(f"number of infs: {np.sum(np.isinf(tomo_data))}")
    #mask_data = mask_in.data

    
    #Normalize the data to have the mean of 0 and std of 1
    tomo_data = (tomo_data - np.mean(tomo_data)) / np.std(tomo_data)        

    tomo_zeros = np.zeros_like(tomo_data)
    print(f"Data shape: {tomo_data.shape}")
    #Apply a median filter on the 0th axis
    #tomo_data = filters.median(tomo_data, mode='reflect')
    #tomo_data = filters.butterworth(tomo_data, 0.1, order=2, npad=32, high_pass=False)
    #tomo_data = filters.laplace(tomo_data, ksize=3)
    #tomo_data = (tomo_data - np.mean(tomo_data)) / np.std(tomo_data)
    tomo_data = (tomo_data - np.min(tomo_data)) / (np.max(tomo_data) - np.min(tomo_data)) 
    # Apply a Sobel filter to the tomogram data

    # Create a figure and axes
    #######################################################################################
    ###USER SELECTION THRESHOLD VALUE ON THE TOMOGRAM##############################
    #######################################################################################
    #threshold_value = popt_granule_fixed[1] +  popt_granule_fixed[2]  # Use the mean of the first gaussian plus two standard deviations as the threshold
    # Mask the tomogram data based on the threshold value
    #tomo_data = np.asanyarray(tomo_data)  # Ensure tomo_data is a numpy array
    threshold_value = 0.5 # Set a default threshold value
    tomo_data_granules = tomo_data < threshold_value #
    current_z_index = 185 # Initialize Z coordinate (image index)

    # Connect the key event to the on_key function
    fig, ax = plt.subplots(figsize=(15, 25))
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.02], facecolor='lightgoldenrodyellow')

    # Create the Slider widget below the x-axis
    plt.subplots_adjust(bottom=0.22)  # Add more space below for the slider
    slider = Slider(ax_slider, 'Threshold', 0.0, 1.0, valinit=threshold_value, valstep=0.01)

    def update_plot(val):
        global threshold_value
        global tomo_data_granules
        threshold_value = slider.val
        tomo_data_granules = tomo_data < threshold_value #
        img_display_granules.set_data(tomo_data_granules[current_z_index,:,:])
        fig.canvas.draw() # Redraw the plot
    slider.on_changed(update_plot)


    def on_key_threshold(event):
        global current_z_index
        global threshold_value
        global tomo_data_granules
        tomo_data_granules = tomo_data < threshold_value #
        #print(f"Z index: {current_z_index}")
        if event.key == 'up':
            current_z_index = min(len(tomo_data) - 1, current_z_index + 1)
        elif event.key == 'down':
            current_z_index = max(0, current_z_index - 1)
        elif event.key == 'enter':
            print(f"Threshold value set to {threshold_value}")
            plt.close()
        for text in ax.texts:
            text.remove()
        #disply the current z index on the image
        ax.text(0.9, 0.95, f"Z index: {current_z_index}", transform=ax.transAxes, color='white', bbox=dict(facecolor='black', alpha=0.5))

        img_display.set_data(tomo_data[current_z_index,:,:])
        img_display_granules.set_data(tomo_data_granules[current_z_index,:,:])
        fig.canvas.draw()
    


    ax.set_title("Press n to reject, any other key to accept")
    # Display the image
    img_display = ax.imshow(tomo_data[current_z_index], cmap='gray')
    # Display the segmented granules
    img_display_granules = ax.imshow(tomo_data_granules[current_z_index], cmap='plasma', alpha=0.3)

    fig.canvas.mpl_connect('key_press_event', on_key_threshold)
    
    plt.title(f"Tomoram Name: {os.path.basename(filename)}")
    plt.show()
    print(f"Threshold value: {threshold_value}")
    plt.close()


    #print("tomo_i incremented to", tomo_i)
    # apply the laplacian filter to the tomogram data
    #tomo_data = 1*(tomo_data > 1) # thresholding the data to remove noise

    #######################################################################################
    ###THRESHOLD AND SEPARATE GRANULES FROM EACH OTHER AND FROM THE BACKGROUND#############
    #######################################################################################
    tomo_data_granules = (tomo_data < threshold_value)*1 # thresholding the data to get granules
    #read the mask file and apply it to the tomogram data
    #maskname = mask_dir + "/" + basename + ".mrc_mask.mrc"
    with mrcfile.open(maskname, mode='r') as mask_in:
        mask_data = mask_in.data
    mask_data = mask_data.astype(np.float32)
    tomo_data_granules = tomo_data_granules * mask_data

    
    #plot the histogram of the granules
    eroded_labels = binary_erosion(tomo_data_granules, ball(2))
    clean_labels = binary_dilation(eroded_labels, ball(2))
    threshold_df.loc[tomo_i,'Tomogram'] = basename
    threshold_df.loc[tomo_i,'Threshold'] = threshold_value
        
    threshold_df.to_csv(f"{basename}_threshold.csv",index=False, header=["Tomogram","Threshold"])
