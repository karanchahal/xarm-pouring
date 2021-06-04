import pickle 
import imageio
states = pickle.load(open('traj/temp_traj.pkl', 'rb'))

frames = []
for s in states:
    rgb = s['rgb'] # 640 x 480 x 3 image
    dep = s['dep'] # 640 x 480 depth image
    xyz_ypr = s['state_cartesian_info'] # x, y,z , yaw, pitch, roll in radians
    state_joint_angles= s['state_joint_angles'] # all 7 joint angles in radians

    frames.append(rgb)

path = "pouring.mp4"
# create video
imageio.mimsave(path, frames, fps=30)