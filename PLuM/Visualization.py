import numpy as np
import open3d as o3d
import os
from utils_plum import R_Intrinsic, read_pc, mesh_coord_frame, play_motion, pyq, Trajectory, homogeneous_intrinsic
from scipy.spatial.transform import Rotation
import numpy as np

PRED_ORT = f'../SIM_LRO_TEST/OLD_WRONG_SCALE/ROTATIONS_PRED_GPU_FT_OPT.npy'
Rec =f'../SIM_LRO_TEST/OLD_WRONG_SCALE/REC_FT/'
SF_DC= 5.720
SF_LR = 2.256
#Rec = "DC/DEPTH_VAL/INT_REC/"
pred = np.loadtxt(PRED_ORT, delimiter=',')
GT = f'../SIM_LRO_TEST/rotations.npy'
gt = np.load(GT)#, delimiter=',')
'''ORT = f'../SIM_DC_TEST/rotations.npy'
gt_ort = np.loadtxt(ORT, delimiter=',')
trans = f'../SIM_DC_TEST/translations.npy'
gt_ort = np.loadtxt(trans, delimiter=',')'''
T = []
pcds = []
pcds_L = []
j = 0
PRED = []
for f in os.listdir(Rec):
    file = os.path.join(Rec, f)
    pts, cen,_ = read_pc(file, sf=SF_LR)
    Q = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcds.append(Q)
    T.append(cen)
    j += 1

'''for d in os.listdir(ORT):
    file = os.path.join(ORT, d)
    pts, cen,_ = read_pc(file)
    R = o3d.io.read_point_cloud(file)
    R.paint_uniform_color([1, 0, 0])
    pcds_L.append(R)


for i in range(len(pcds)):
    o3d.visualization.draw_geometries([pcds[i], pcds_L[i], mesh_coord_frame])
'''
PRED = []
for i in range(len(pred)):
    pose = np.eye(4, 4)
    A = pred[i]
    #t = T[i]
    #A = Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_euler("XYZ", degrees=True)
    ort = o3d.io.read_point_cloud('LRO_test/LRO_cen_2048.pcd')
    #DC/files/DC_2048_cen.pcd
    pose = homogeneous_intrinsic(A[0], A[1], A[2], 0,0,0)
    # ort = ort.farthest_point_down_sample(512)
    #ort = ort.transform(pose)
    #pose1 = homogeneous_intrinsic(0,-90,0, 0,0,0)
    #pose = pose1 @ pose
    ort = ort.transform(pose)
    ort.paint_uniform_color([1, 0, 0])
    PRED.append(ort)


for j in range(len(gt)):
    pose = np.eye(4, 4)
    q = gt[j]
    #t = T[i]
    A = Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_euler("XYZ", degrees=True)
    ort = o3d.io.read_point_cloud('LRO_test/LRO_cen_2048.pcd')
    #DC/files/DC_2048_cen.pcd
    pose = homogeneous_intrinsic(A[0], A[1], A[2], 0,0,0)
    pose1 = homogeneous_intrinsic(0,0,90, 0,0,0)
    # ort = ort.farthest_point_down_sample(512)
    #ort = ort.transform(pose1)
    pose = pose @ pose1
    ort = ort.transform(pose)
    ort.paint_uniform_color([0, 0, 0])
    pcds_L.append(ort)

user_input = input('show video pcd ?(yes/no): ')
if user_input.lower() == 'yes':
    play_motion(PRED, pcds_L)

elif user_input.lower() == 'no':
    print('user typed no')



'''
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection='3d')
ax.set_aspect('equal')

sc1 = ax.scatter([], [], [], s=10, color='red', label="Predicted Pose")
sc2 = ax.scatter([], [], [], s=10, color='black', label="Target Pose")
bar_ax = fig.add_axes([0.75, 0.2, 0.2, 0.6])  # [left, bottom, width, height] in figure coordinates


def add_bar_chart(ax, bar_height):
    # Position the bar chart
    bar_x = [3]
    bar_y = [3]
    bar_z = [-1]
    bar_dx = [0.1]  # Width
    bar_dy = [0.1]  # Depth
    bar_dz = [bar_height]  # Height

    # Clear previous bars and draw new ones
    ax.bar3d(bar_x, bar_y, bar_z, bar_dx, bar_dy, bar_dz, color='blue', alpha=0.8)


# Update function for animation


def update(frame):
    # print(np.shape(PRED[frame]))
    x1, y1, z1 = PRED[frame].T
    x2, y2, z2 = pcds[frame].T

    # Update the two scatter plots
    sc1._offsets3d = (x1, y1, z1)
    sc2._offsets3d = (x2, y2, z2)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([-3, 3])
    ax.legend(loc="upper right")
    bar_height = np.abs(np.sin(frame * 0.1)) * 10  # Dynamic bar height
    bar_ax.clear()
    bar_ax.bar([0], [bar_height], color='blue', width=0.5)
    bar_ax.set_ylim(0, 10)
    bar_ax.set_xticks([])  # Remove x-axis ticks for simplicity
    bar_ax.set_yticks([0, 5, 10])  # Set y-axis ticks
    bar_ax.set_title("Dynamic Bar")
    # ax.view_init(elev=0, azim=0, roll=0)

    return sc1, sc2


# Create animation
num_frames = 700
# ax.view_init(elev=0, azim=0, roll=90)
# Save animation as a video
ani = FuncAnimation(fig, update, frames=num_frames, interval=20, blit=False)
writergif = anim.PillowWriter(fps=35)
ani.save('pose_2.gif', writer=writergif)  # Save animation as a video

# Display the animation
plt.show()'''

'''user_input = input('show video pcd ?(yes/no): ')
if user_input.lower() == 'yes':
    play_motion(PRED, pcds)

elif user_input.lower() == 'no':
    print('user typed no')
'''
