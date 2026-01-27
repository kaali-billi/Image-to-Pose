from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from utils_plum import readLookupTable,read_pc
import random
#T = [2.58,2.00,1.75]
T = [2.48,2.01,1.77]
pts, _,_ = read_pc('LRO_test/LRO_cen_2048.pcd', sf = 2.713)
t = np.expand_dims(np.array(T), axis=0)
pts = pts.T + t.T
print(np.shape(pts))
c1,c2,c3 = pts


lookup = readLookupTable('LRO_test/Reward_222_020mm_sig01.txt')
stepSize = 0.02
maxXYZ = [5,5,4]

numX = round(maxXYZ[0] / stepSize) + 1
numY = round(maxXYZ[1] / stepSize) + 1
numZ = round(maxXYZ[2] / stepSize) + 1

xDim = np.linspace(0, maxXYZ[0], numX)
yDim = np.linspace(0, maxXYZ[1], numY)
zDim = np.linspace(0, maxXYZ[2], numZ)
points_per_meter = 1 / stepSize
num_xyz = [round(maxXYZ[i] * points_per_meter + 1) for i in range(3)]

Reward = []
coord = []


for x in tqdm(xDim):
    for y in yDim:
        for z in zDim:
            x_index = round(x * points_per_meter)
            y_index = round(y * points_per_meter)
            z_index = round(z * points_per_meter)
            index = z_index + (y_index * num_xyz[2]) + (x_index * num_xyz[1] * num_xyz[2])
            #if lookup[index] >= 250:
            Reward.append(lookup[index])
            coord.append([x,y,z])
import random



random.seed(42)
Reward = random.sample(np.array(Reward),126632)
coord = random.sample(np.array(coord),126632)

x, y, z = coord.T
norm = plt.Normalize(Reward.min(), Reward.max())
colormap = plt.cm.viridis
colors = colormap(norm(Reward))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
sc1 = ax.scatter(x,y,z, s=1, color=colors, label="Lookup Table")
sc2 = ax.scatter(c1,c2,c3, s=5, color='red', label="Target")
#scatter = ax.scatter3D(x,y,z, c=colors, marker='o', s=2)
cbar = plt.colorbar(sc1, ax=ax)
cbar.set_label('Reward')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.set_xlim(0,10)
ax.set_ylim(0,10)
ax.set_zlim(0,10)
#fig.savefig('lookupimp.png')  # Save as PNG image
plt.show()