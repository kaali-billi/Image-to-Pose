import copy
from tqdm import tqdm
import numpy as np, math, open3d as o3d
from utils_plum import homogeneous_intrinsic


# .stl file path
fileName = "DC/files/Dream Chaser.stl"

lookupToModel = homogeneous_intrinsic(0, 0, 0, 2,2,2) # Ensure all aprts of structure are in postive X,Y,Z
stepSize = 0.1 # Fidelity of reward table, step size = 0.1 for dimension of 1m means 10 ponits sampled between 0 and 1 m.
maxXYZ = [5,5,4]  # [x,y,z] # Max Dimension of Object in all X,Y,Z
sigma = 0.1
# load mesh and convert to open3d.t.geometry.TriangleMesh
mesh = o3d.io.read_triangle_mesh(fileName)
mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

# minimum and maximum geometry bounds
min_bound = mesh.vertex.positions.min(0).numpy()
max_bound = mesh.vertex.positions.max(0).numpy()
print("min bound : ", min_bound)
print("max bound : ", max_bound)

# transform the mesh to the correct frame
mesh_t = copy.deepcopy(mesh).transform(lookupToModel)

# Create a scene and add the triangle mesh
scene = o3d.t.geometry.RaycastingScene()
_ = scene.add_triangles(mesh_t)  # we do not need the geometry ID for mesh

# updated minimum and maximum geometry bounds
min_bound = mesh_t.vertex.positions.min(0).numpy()
max_bound = mesh_t.vertex.positions.max(0).numpy()
print("min bound : ", min_bound)
print("max bound : ", max_bound)
print(mesh_t.get_center())
query_point = o3d.core.Tensor([[0, 0, 0]], dtype=o3d.core.Dtype.Float32)

# print(query_point.numpy())
signed_distance = scene.compute_signed_distance(query_point)
print("signed Distance : ", signed_distance.numpy())

closestDistance = []
closestOccupancy = []
rewardValues = []

numX = round(maxXYZ[0] / stepSize) + 1
numY = round(maxXYZ[1] / stepSize) + 1
numZ = round(maxXYZ[2] / stepSize) + 1

xDim = np.linspace(0, maxXYZ[0], numX)
yDim = np.linspace(0, maxXYZ[1], numY)
zDim = np.linspace(0, maxXYZ[2], numZ)
i = 0
j = 0
coord = []
for x in tqdm(xDim):
    for y in yDim:
        for z in zDim:
            query_point = o3d.core.Tensor([[x, y, z]], dtype=o3d.core.Dtype.Float32)
            # print(query_point.numpy(), i)
            # j += 1
            signed_distance = scene.compute_signed_distance(query_point)
            if signed_distance >= 0:
                i += 1
                reward = 255 * math.exp(-0.5 * signed_distance.numpy() * signed_distance.numpy() / (sigma * sigma))
                rewardValues.append(reward)
            else:
                rewardValues.append(0)





rewardsToWrite = np.array(rewardValues, dtype=np.uint8)

# SAVE REWARDS FILE/ Lookup File
np.savetxt('LRO_test/Reward_LRO_100mm_sig01.txt', rewardsToWrite, delimiter=',')

'''with open(f'LRO_test/CONFIG_010mm_sig01.txt', 'w') as f:
    # Write variable names as plain text without the comment symbol
    f.write('Sigma: ')
    np.savetxt(f, [sigma], fmt='%.2f', delimiter=',')
    f.write('StepSize: ')
    np.savetxt(f, [stepSize], fmt='%.2f', delimiter=',')
    f.write('maxXYZ: ')
    np.savetxt(f, maxXYZ, fmt='%.2f', delimiter=',')
    f.write('lookup: ')
    np.savetxt(f, lookupToModel, fmt='%.2f', delimiter=',')
    f.write('EVD_trans: ')
'''
