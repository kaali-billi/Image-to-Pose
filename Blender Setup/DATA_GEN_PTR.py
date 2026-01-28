import os
import open3d as o3d
from tqdm import tqdm
import numpy as np
from UTILS import extract_data_PTR
from PLuM_Python.utils_plum import *
import pyquaternion as pyq


path_npz = 'PLuM_Python/LRO_test/NPZ'
pcd = 'Space_Shuttle_data/Dream_chaser.pcd'
path_exr = 'Space_Shuttle_data/EXR_PTR/'
object = 'LRO_35'
EXR = []
orts = []

for file in tqdm(sorted(os.listdir(path_npz))):
    f = os.path.join(path_npz, file)
    Q,t = extract_data_PTR(f,object)
    r = Rotation.from_quat([Q[1],Q[2],Q[3],Q[0]]).as_euler(SEQ,True)
    orts.append(r)

np.save('PLuM_Python/LRO_test/GT_ORTS.npy', orts)

