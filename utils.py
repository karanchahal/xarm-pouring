import numpy as np
import cv2
import matplotlib.pyplot as plt 
from xarm.wrapper import XArmAPI
import imageio
import glob
import os
import pickle
import pyrealsense2 as rs


def get_angles_from_xarm_studio_traj(l):
    l = l.decode()
    nums = l.split(',')
    integer_map = map(float, nums[:-1])
    integer_list = list(integer_map)
    return integer_list

def get_img_and_depth(realsense, align, hole_filling, vis=True, height=640, width=480, crop=False):
    """
    Img is in uint
    Depth is in milimeters
    """

    frames = realsense.wait_for_frames()
        
    aligned_frames = align.process(frames)


    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    depth_frame = hole_filling.process(depth_frame)

    if not depth_frame or not color_frame:
        print("ERROR: no new images receieved !")
        return

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    if crop:
        color_image = color_image[:-100, 100:550].astype(np.uint8)
        depth_image = depth_image[:-100, 100:550].astype(np.float32)


    if vis:
        # cv2.imshow('RealSense', images)
        cv2.imshow('frame', cv2.resize(color_image, (height, width)))
        # plt.imshow(depth_image)
        # plt.show()
        cv2.imshow('frame2', cv2.resize((depth_image - depth_image.min()) / (depth_image.max() - depth_image.min()), (height, width)))
        cv2.waitKey(1)

    return cv2.resize(color_image, (height, width)), cv2.resize(depth_image, (height, width))


def get_arm(ip='192.168.1.246'):
    arm = XArmAPI(ip)
    arm.motion_enable(enable=True)
    arm.set_mode(0)
    arm.set_state(state=0)

    return arm

def create_video(dataset_path, path):
    list_of_files = glob.glob(os.path.join(dataset_path, '*')) # * means all if need specific format then *.csv
    frames = []
    for f in list_of_files:
        obj = pickle.load(open(f, 'rb'))
        frames.append(obj['img'])
    
    imageio.mimsave(path, frames, fps=10)


def create_video_from_file(file_path, path):
    frames = []
    obj = pickle.load(open(file_path, 'rb'))
    for o in obj:
        frames.append(o['rgb'])
    imageio.mimsave(path, frames, fps=10)

def create_video_from_frames(frames, path):
    imageio.mimsave(path, frames, fps=10)

def get_all_files(folder):
    list_of_files = glob.glob(os.path.join(folder, '*')) # * means all if need specific format then *.csv
    return latest_file


class Rate:
    def __init__(self, period):
        self._period = period
        self._last = time.time()

    def sleep(self):
        delta = time.time() - self._last
        time.sleep(max(0, self._period - delta))
        self._last = time.time()

def _start_realsense():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

    # Start streaming
    pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    hole_filling = rs.hole_filling_filter()

    for x in range(5):
        pipeline.wait_for_frames()


    return pipeline, align, hole_filling


def move_arm(arm, is_radian, x, y, z, roll, pitch, yaw, wait=True):
    '''
    Moves end effector to mentioned position
    '''
    code = arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, is_radian=is_radian, wait=wait)
    if code != 0:
        print(f"Bad code is {code}")
    assert code == 0


def set_joint_angles(arm, angles, is_radian, wait=True):
    '''
    angles should be a list of 7 float values signifying the joint angle position.

    Change wait=True for smoother movements, (for PD control)
    '''
    code = arm.set_servo_angle(angle=joint, wait=wait, is_radian=is_radian)
    if code != 0:
        print(f"Bad code is {code}")
    assert code == 0


def _get_state(arm, is_radian=True):
    state = {}

    # position
    code, curr_pos = arm.get_position(is_radian=is_radian)
    state['position'] = curr_pos

    code, angle_list = arm.get_servo_angle(is_radian=is_radian)
    state['servo_angle_list'] = angle_list

    code, grip_pos = arm.get_gripper_position()
    state['gripper_pos'] = grip_pos

   
    return state

def reset(arm):
    """
    Reset the robot
    """
    arm.reset(wait=True)

def step(arm, x, y, z, wait=True):
    """
    This function steps in the x,y,z directions relative to the current position
    """
    code = arm.set_position_aa([x, y, z, 0, 0, 0], relative=True, is_radian=False, wait=wait)
    # get observation: if the robot is set to wait=True, then this observation is valid, other wise, we should do something else.

    assert code == 0
    
    return state