import pickle 
import imageio

cup_type = 'red_cup'
position = 1
states = pickle.load(open(f'pouring_training_trajs/{cup_type}/{position}.pkl', 'rb'))

for s in states:
    '''
    Keys in dict of s are:
    1. rgb
    2. depth
    3. state_joint_angles of the robot currently
    4. end effector position (x, y, z , yaw , pitch , roll)
    5. actions: joint angle position that the robot should take next
    '''
    print(s.keys())
    exit()