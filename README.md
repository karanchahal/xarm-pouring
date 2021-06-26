# Getting Trajectories

Go to [this link](https://drive.google.com/drive/folders/1Zm4u-kaTnvkoQ1pWkKsCA80pFGZPQxwy?usp=sharing) to get pouring trajectories.

Place the trajectory pickle files in the ```./trajs``` folder at the root of this project

The files to access the pretext data is ```pouring_dataloader.py```

# Getting Pretext Task Dataset

Go to [this link](https://drive.google.com/drive/folders/1XDeOF_zGl4GVvH0oMIhB6Caw4-pv3YcZ?usp=sharing) to get data for the pretext task.

The files to access the pretext data is ```pretext_task_dataloader.py```

# Robot Infra

A few useful files that would help interfacing with the robots. Hopefully, we can build this up to have first class I/O and Teleoperation support.

## Collecting Trajectories

1. Download xARM Studio.
2. Attach the gripper of the robot and initilize it's initial position. 
3. Go to recordings and collect a trajectory.
4. Download a suitable trajectory.

Note on collecting a trajectory:

1. Mark 15 positions on the table.
2. Put a cup at that position
3. Add a cup to the gripper of the robot and put almonds in it.
4. Manually take the robot and do the pouring.
5. Save the recording.
6. xARM studio has a very easy to use UI for this. 

## Parsing Trajectory

Now, we would like to collect image observations with those trajectories.
Take the trajectory and extract it in the ```./trajs/``` folder. The format of the traejctories for pouring should be as follows:
- ```trajs/{cup_type}/{cup_type}_pos_{num}.traj``` where cup type represents the type of cup and num represents the trajectory number.

1. Now run ```python parse_traj.py --cup-type <red_cup/plastic_cup...> --num <1-15> ``` which will take the trajectory and parse it and save the parsed trajectory in ```./parsed_trajs/<name_of_traj_file>```

2. This might take a while as we collect image observation for each action.


The parsed trejctory is of this data type:

1. ```List[Dict]```, where the dict has the following keys:
``` 
{
'rgb': Image observation
'dep': Depth observation,
'state_joint_angles': The joint angles of the xarm robot currently,
'state_cartesian_info': The x, y,z , yaw pitch and roll of the robot.
'actions': The actions i.e the joint angles that the robot should go to next.
}
```
## Running a policy on the robot

1. Configure the robot state.
2. Run ```deploy_policy.py --expID 200x --algo <mlp/ndp>``` to deploy a certain policy. The expID represents the policy which is meant to be deployed and the algo argument represents whether the policy is a MLP or an NDP.


## Useful Functions

All utils functions are in ```utils.py```

### Realsense Camera stuff

1. init realsense camera: ```_start_realsense()```
2. Get image and depth observations: ```get_img_and_depth()```
3. init xarm: ```get_arm()```, gives you an object which can be used to control the robot.

### Robot end effector control

1. ```step()```: moves the end effector by a delta x, y and z
2. ```move()```: moved the end effector to a particular x,y,z, yaw, pitch, roll
3. ```set_joint_angles()```: sets the robot into a specific joint angle configuration (the xarm has 7 joints, so 7 values)
4. ```_get_state()```


