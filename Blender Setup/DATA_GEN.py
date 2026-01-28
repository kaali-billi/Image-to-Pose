"""
DESCRIPTION : Pipeline for DATASET GENERATION from Extracted Depth(EXR) files and
              GT Orientation(NPZ) files
"""

import os
from sklearn.model_selection import train_test_split
from UTILS import create_test_test, extract_data_PTR
import matplotlib.pyplot as plt

import open3d as o3d
from tqdm import tqdm
import time
import numpy as np



import numpy as np
from depth_data.Depth_utils import depth_map, overlay
from sklearn.model_selection import train_test_split
import os
import cv2 as cv
from tqdm import tqdm

img1 = 'SIM_LRO_TEST/IMG/'
npz1 = 'SIM_LRO_TEST/NPZ/'
depth1 = 'SIM_LRO_TEST/DEPTH/'


images = []
depths = []
masks = []
fl = 0
object = "foil_silver"
rots = []
Translation = []


for file in tqdm(os.listdir(img1)):
    f = os.path.join(img1, file)
    t = file.split('Image')[-1].split('.png')[0]
    npz_file = t + ".npz"
    NPZ_file = os.path.join(npz1, npz_file)
    depth_file = "Depth" + t + ".exr"
    img, msk = overlay(f, NPZ_file)

    images.append(img)
    masks.append(msk)
    depths.append(os.path.join(depth1, depth_file))

    orts, T = extract_data_PTR(NPZ_file, object)
    rots.append(orts)
    Translation.append(T)






'''    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Display the original image
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Display the segmentation mask
    axes[1].imshow(msk, cmap='jet')  # Using 'jet' colormap for segmentation
    axes[1].set_title('Segmentation Map')
    axes[1].axis('off')

    # Display overlay (image with mask overlay)
    axes[2].imshow(img)
    axes[2].imshow(msk, alpha=0.5, cmap='jet')  # Semi-transparent overlay
    axes[2].set_title('Image with Overlay')
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
    fl+=1
    if fl == 5:
        break'''

'''
for file2 in tqdm(os.listdir(img2)):
    q = os.path.join(img2, file2)
    z = file2.split('Image')[-1].split('.png')[0]
    npz_file = z + ".npz"
    depth_file = "Depth" + z + ".exr"
    img, msk = overlay(q, os.path.join(npz2, npz_file))
    images.append(img)
    masks.append(msk)
    depths.append(os.path.join(depth2, depth_file))

for dile in tqdm(os.listdir(img3)):
    d = os.path.join(img3, dile)
    r = dile.split('Image')[-1].split('.png')[0]
    npz_dile = r + ".npz"
    depth_dile = "Depth" + r + ".exr"
    img1, msk1 = overlay(d, os.path.join(npz3, npz_dile))
    images.append(img1)
    masks.append(msk1)
    depths.append(os.path.join(depth3, depth_dile))'''

print("All Files Collected")
print("Train Files:", len(images))
ROTS = np.array(rots)
TRAN = np.array(Translation)
#assert len(images) == len(depths) == len(masks) == len(rots) == len(Translation)

'''test_size = 0.0
if test_size > 0:
    train_images, val_images, train_masks, val_masks, train_depths, val_depths = train_test_split(
        images,
        masks,
        depths,
        test_size=test_size,  # 20% for validation
        random_state=42,  # For reproducibility
    )
    print(len(train_images), len(val_images))

else:

    train_images, train_masks, train_depths = images, masks, depths
    print(len(train_images))'''

train_images, train_masks, train_depths = images, masks, depths
t = 0
non = 0
valid_rots = []
valid_trans = []
for i in tqdm(range(len(train_depths)), "Training Set"):
        dpt = depth_map(train_depths[i], 0)
        dpt_min = np.min(dpt[dpt != -1])
        if 0 < dpt_min < 35:
            cv.imwrite('SIM_LRO_TEST/test_img/{}.png'.format(str(t).zfill(4)), train_images[i])
            cv.imwrite('SIM_LRO_TEST/test_msk/{}.png'.format(str(t).zfill(4)), train_masks[i])
            np.save('SIM_LRO_TEST/test_depth/{}.npy'.format(str(t).zfill(4)),dpt)  # depth_map(depths[i],0)
            valid_rots.append(ROTS[i])
            valid_trans.append(TRAN[i])
            t += 1
        else:
            non += 1


print(t,non)
np.save('SIM_LRO_TEST/rotations_CORR.npy', np.array(valid_rots))
np.save('SIM_LRO_TEST/translations.npy', np.array(valid_trans))



