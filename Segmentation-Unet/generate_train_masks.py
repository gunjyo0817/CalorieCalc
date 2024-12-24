import os
import xml.etree.ElementTree as ET
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tifffile import imwrite
from matplotlib import font_manager

# Create output directory
output_dir = "./data/train_masks"
os.makedirs(output_dir, exist_ok=True)

# Load XML file
# tree = ET.parse('./data/training_data_final.xml')
tree = ET.parse('./data/training_data_final.xml')
root = tree.getroot()

# Define label to channel mapping based on XML file
labels = root.find(".//labels")
label_map = {label.find("name").text: idx + 1 for idx, label in enumerate(labels)}

# Process each image
for image in root.findall('.//image'):
    width = int(image.get('width'))
    height = int(image.get('height'))
    image_name = image.get('name').split('.')[0]
    
    # Initialize an empty mask with 20 channels
    mask = np.zeros((height, width, 37), dtype=np.uint8)
    
    # Draw each polygon in its respective channel
    for polygon in image.findall('.//polygon'):
        label = polygon.get('label')
        channel = label_map[label]  # Get the channel for this label
        
        # Get points and draw polygon on a 2D slice of the mask
        points = [tuple(map(lambda x: int(float(x)), p.split(','))) for p in polygon.get('points').split(';')]
        single_channel_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(single_channel_mask, [np.array(points)], 1)
        
        # Add the filled single channel mask to the appropriate channel in the mask
        mask[:, :, channel] = single_channel_mask
    
    # Save the mask as a TIFF file in "train_masks" folder
    tif_path = os.path.join(output_dir, f"{image_name}.tif")
    imwrite(tif_path, mask)  # Changed from imsave to imwrite
    
    # Plot the mask for visualization
    plt.figure(figsize=(10, 10))
    plt.title(f"One-Hot Encoded Mask - {image_name}")
    plt.imshow(mask.sum(axis=2), cmap='gray')  # Summing across channels for visualization
    plt.axis('off')
    
    # Save plot as PNG in the same folder
    # plot_path = os.path.join("train_masks", f"{image_name}_mask_plot.png")
    # plt.savefig(plot_path)
    plt.close()
