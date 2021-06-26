import numpy as np 

from utils import get_img_and_depth, get_arm, _start_realsense
import cv2
import matplotlib.pyplot as plt
import time 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from a2c_ppo_acktr.utils import init
import numpy as np
import os
import time
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from robot_benchmark.imednet.imednet.utils.dmp_layer_obj import DMPIntegrator, DMPParameters
from torchvision import transforms 
import argparse
import cv2
import pyrealsense2 as rs
cap = cv2.VideoCapture(0)
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--N', type=int, default=220)
parser.add_argument('--T', type=int, default=249)
parser.add_argument('--l', type=int, default=1)
parser.add_argument('--az', type=int, default=15)
parser.add_argument('--scale', type=int, default=5)
parser.add_argument('--expID', type=int, default=2003)
parser.add_argument('--axis', type=str, default='pos', help='pos|type|full')
parser.add_argument('--algo', type=str, default='ndp', help='ndp|mlp')
args = parser.parse_args()

log_dir = './'  + str('{:05d}'.format(args.expID))
print(f"Loading model from {log_dir}")
args = parser.parse_args()
normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
class Net(nn.Module):
    def __init__(self, bias=None):
        super(Net, self).__init__()
        c1, a1 = nn.Conv2d(3, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True)
        c2, a2 = nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(inplace=True)
        m1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        c3, a3 = nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.ReLU(inplace=True)
        c4, a4 = nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.ReLU(inplace=True)
        self.vgg = nn.Sequential(c1, a1, c2, a2, m1, c3, a3, c4, a4)

        self.extra_convs = nn.Conv2d(128, 128, 3, stride=2, padding=1)

        fc1, a1 = nn.Linear(256, 16), nn.ReLU(inplace=True)
        fc2 = nn.Linear(16, 3, bias=False)
        self.top = nn.Sequential(fc1, a1, fc2)
        bias = np.zeros(3).astype(np.float32) if bias is None else np.array(bias).reshape(3)
        self.register_parameter('bias', nn.Parameter(torch.from_numpy(bias).float(), requires_grad=True))

    def forward(self, x):
        # vgg convs and 2D softmax
        x = self.vgg(x)
        x = self.extra_convs(x)
        B, C, H, W = x.shape
        x = F.softmax(x.view((B, C, H * W)), dim=2).view((B, C, H, W))

        # find expected keypoints
        h = torch.linspace(-1, 1, H).reshape((1, 1, -1)).to(x.device) * torch.sum(x, 3)
        w = torch.linspace(-1, 1, W).reshape((1, 1, -1)).to(x.device) * torch.sum(x, 2)
        x = torch.cat([torch.sum(a, 2) for a in (h, w)], 1)

        # regress final pose and add bias
        x = self.top(x) + self.bias
        return x
    def forward_traj(self, x):
        x = self.vgg(x)
        x = self.extra_convs(x)
        B, C, H, W = x.shape
        x = F.softmax(x.view((B, C, H * W)), dim=2).view((B, C, H, W))

        # find expected keypoints
        h = torch.linspace(-1, 1, H).reshape((1, 1, -1)).to(x.device) * torch.sum(x, 3)
        w = torch.linspace(-1, 1, W).reshape((1, 1, -1)).to(x.device) * torch.sum(x, 2)
        x = torch.cat([torch.sum(a, 2) for a in (h, w)], 1)
        return x

class CNN(nn.Module):
    def __init__(
            self,
            T=250,
            hidden_activation=F.tanh,
            freeze=False,
            *args,
            **kwargs):
        super().__init__()
        self.T = T
        self.output_dim = T*7
        output_size = T*7
        self.hidden_activation = hidden_activation
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.fc1 = init_(nn.Linear(256, output_size//2))
        self.layers = [self.fc1]
        self.fc_last = init_(nn.Linear(output_size//2, output_size))
        self.pt = net
        if freeze: 
            for param in self.pt.parameters():
                param.requires_grad = False


    def forward(self, input, state=None):
        h = self.pt.forward_traj(input)
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        output = self.fc_last(h)
        return output.reshape(-1, self.T, 7)

class DMPNet(nn.Module):
    def __init__(
            self,
            hidden_size=256,
            hidden_activation=F.tanh,
            N = 5,
            T = 10,
            l = 10,
            tau = 1,
            goal_type='int_path',
            rbf='gaussian',
            num_layers=1,
            a_z=15,
            state_index=np.arange(7),
            freeze=False,
            *args,
            **kwargs):
        super().__init__()
        self.N = N
        self.l = l
        self.goal_type = goal_type
        self.vel_index = vel_index
        self.output_size = N*len(state_index) + len(state_index)
        output_size = self.output_size
        dt = tau / (T*self.l)
        self.T = T
        self.output_activation=torch.tanh
        self.DMPparam = DMPParameters(N, tau, dt, len(state_index), None, a_z=a_z)
        self.func = DMPIntegrator(rbf=rbf, only_g=False, az=False)
        self.register_buffer('DMPp', self.DMPparam.data_tensor)
        self.register_buffer('param_grad', self.DMPparam.grad_tensor)
        self.state_index = state_index
        self.output_dim = output_size
        self.hidden_activation = hidden_activation
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        if num_layers > 1:
            self.fc1 = init_(nn.Linear(40, hidden_size))
            self.fc2 = init_(nn.Linear(hidden_size, output_size//2))
            self.layers = [self.fc1, self.fc2]
            self.fc_last = init_(nn.Linear(output_size//2, output_size))
        else:
            self.fc_last = init_(nn.Linear(256, output_size))
            self.layers = []
        self.pt = net
        if freeze: 
            for param in self.pt.parameters():
                param.requires_grad = False
   
    def forward(self, input, state=None, vel=None, return_preactivations=False, first_step=False):
        h = self.pt.forward_traj(input)
        for layer in self.layers:
            h = self.hidden_activation(layer(h))
        output = self.fc_last(h)*args.scale
        y0 = state[:, self.state_index].reshape(input.shape[0]*len(self.state_index))
        dy0 = torch.ones_like(y0)*0.05
        # dy0 = velreshape(input.shape[0]*len(self.state_index))
        y, dy, ddy = self.func.forward(output, self.DMPp, self.param_grad, None, y0, dy0)
        y = y.view(input.shape[0], len(self.state_index), -1)
        y = y[:, :, ::self.l]
        return y.transpose(1, 2)


net = Net()
net = net.eval()

if args.algo == 'ndp': 
    T = args.T
    N = args.N
    l = args.l
    a_z = args.az
    hidden_sizes = 100
    state_index = np.arange(7)
    vel_index = None
    dmpn = DMPNet(N=N, l=l, T=T, a_z=a_z, hidden_size=hidden_sizes,state_index=state_index, vel_index=vel_index)

else: 
    dmpn = CNN()


dmpn.load_state_dict(torch.load(log_dir + '/policy.pt', map_location='cpu'))


def proc_frame(img):
    data = cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA)
    data = data.astype(np.float32) / 255
    data -= np.array([0.485, 0.456, 0.406]).reshape((1, 1, 3))
    data /= np.array([0.229, 0.224, 0.225]).reshape((1, 1, 3))
    return data[None].transpose((0, 3, 1, 2))

arm = get_arm()
realsense, align, hole_filling = _start_realsense()

for i in range(100):
    # take the last image because the realsense sometimes takes time to adjust colors
    img, depth = get_img_aqd_depth(realsense, align, hole_filling)


img = proc_frame(img)
err_code, curr_joint_angles = arm.get_servo_angle(is_radian=True)
arm.set_servo_angle(angle=curr_joint_angles, is_radian=True, wait=True)
robot_state = torch.Tensor(curr_joint_angles).view(1,7)

img = torch.Tensor(img)


traj = dmpn(img, state=robot_state).detach().numpy()[0]

for joint in traj:
    time.sleep(0.03)
    arm.set_servo_angle(angle=joint, wait=False, is_radian=True)
    input("click any button to go to next state")
