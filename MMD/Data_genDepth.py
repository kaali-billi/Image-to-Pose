import numpy as np
from Depth_utils import depth_map, overlay
from sklearn.model_selection import train_test_split
import os
import cv2 as cv
from tqdm import tqdm


# Directories for images, depths and masks
img1 = 'LRO/i1/'
img2 = 'LRO/i2/'
img3 = 'LRO/i3/'
npz1 = 'LRO/n1/'
npz2 = 'LRO/n2/'
npz3 = 'LRO/n3/'
depth1 = 'LRO/d1/'
depth2 = 'LRO/d2/'
depth3 = 'LRO/d3/'


images = []
depths = []
masks = []

for file in tqdm(os.listdir(img1)):
    f = os.path.join(img1, file)
    t = file.split('Image')[-1].split('.png')[0]
    npz_file = t + ".npz"
    depth_file = "Depth" + t + ".exr"
    img, msk = overlay(f, os.path.join(npz1, npz_file))
    images.append(img)
    masks.append(msk)
    depths.append(os.path.join(depth1, depth_file))
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
assert len(images) == len(depths) == len(masks)


# Train Test Split
test_size = 0.15
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
    print(len(train_images))

t = 0
non = 0
for i in tqdm(range(len(train_images)), "Training Set"):
        dpt = depth_map(train_depths[i], 0)
        dpt_min = np.min(dpt[dpt != -1])
        if dpt_min > 0:
            # print(dpt_min)
            cv.imwrite('LRO/Train/images/{}.png'.format(str(t).zfill(4)), train_images[i])
            cv.imwrite('LRO/Train/masks/{}.png'.format(str(t).zfill(4)), train_masks[i])
            np.save('LRO/Train/depths/{}.npy'.format(str(t).zfill(4)),
                    depth_map(train_depths[i], 0))  # depth_map(depths[i],0)
            t += 1
        else:
            non += 1

print("Invalid Train Files :", non)
nont = 0
if test_size > 0:
    k = 1917
    for j in tqdm(range(len(val_images)), "Validation Set"):
        dpt = depth_map(val_depths[j], 0)
        dpt_min = np.min(dpt[dpt != -1])
        if dpt_min > 0:
            cv.imwrite('LRO/Val/images/{}.png'.format(str(k).zfill(4)), val_images[j])
            cv.imwrite('LRO/Val/masks/{}.png'.format(str(k).zfill(4)), val_masks[j])
            np.save('LRO/Val/depths/{}.npy'.format(str(k).zfill(4)), depth_map(val_depths[j], 0))
            k += 1
        else:
            nont += 1

print("Invalid test Files :", nont)
