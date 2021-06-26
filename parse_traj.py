import os
import sys
import time
import math
from utils import get_img_and_depth, get_arm, get_angles_from_xarm_studio_traj, Rate, create_video_from_file, _start_realsense
sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
import pickle 
import argparse
from xarm.wrapper import XArmAPI

parser = argparse.ArgumentParser()

parser.add_argument('--cup-type', type=str)
parser.add_argument('--num', type=int)
parser.add_argument('--sample-rate', type=int, default=100)

args = parser.parse_args()

cup_type = args.cup_type
num = args.num

sample_rate = args.sample_rate # skip 100 observations to make parsing trejctory faster, can be toggled
states = []
arm = get_arm(ip)

# init image stuff
realsense, align, hole_filling = _start_realsense()

# open xarm trajectory
f = open(f'trajs/{cup_type}/{cup_type}_pos_{num}.traj', 'rb')
Lines = f.readlines()


for idx, l in enumerate(Lines):
    if idx % int(sample_rate) == 0:

        angles = get_angles_from_xarm_studio_traj(l)
        # get img obs
        rgb, dep = get_img_and_depth(realsense, align, hole_filling)
        # get current joint angles
        current_joint_angles = arm.get_servo_angle(is_radian=True)
        # get cartesian information
        cartesian_info = arm.get_position(is_radian=True)

        arm.set_servo_angle(angle=angles, wait=True, is_radian=True)
        state = {
            'rgb': rgb,
            'dep': dep,
            'state_joint_angles': current_joint_angles,
            'state_cartesian_info': cartesian_info, # x, y, z, yaw, pitch, roll
            'actions': angles
        }

        states.append(state)


pickle.dump(states, open(f'parsed_trajs/{cup_type}/{num}.pkl', 'wb'))
arm.disconnect()

# record video and save

create_video_from_file(file_path=f'parsed_trajs/{cup_type}/{num}.pkl', path=f'parsed_trajs/videos/{cup_type}/{num}.mp4')
