#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
from casadi import *
from casadi.tools import *
import pdb
import sys
import time
sys.path.append('../../')
sys.path.append("../utils/")
sys.path.append("scripts/")
sys.path.append('../../')
sys.path.append("../../social_nav/datatext/")
sys.path.append("../../social_nav/saved_models/")
sys.path.append("../../social_nav/scripts/")
sys.path.append("../../social_nav/utils/")
import do_mpc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader
import argparse

from template_model import template_model
from template_mpc import * #template_mpc
from template_simulator import template_simulator

from pred_egonet import*

import socket
import ros
import rospy
from digit_main.msg import MPCState # if this import doesn't work, try source ~/<CATKIN_WORKSPACE>/devel/setup.bash
                                    # Also, make sure that Digit's code has been catkin_made with the MPC branch so that this message exists
from digit_main.msg import PredictionState

""" User settings: """
show_animation = False
store_results = True


obstacles = [
    {'x': 5, 'y': 8.5, 'r':0.2, 'x2': 8, 'y2': 8.8, 'r2': 0.2},
]


"""
Get configured do-mpc modules:
"""
model = template_model()
mpc = template_mpc(model)
simulator = template_simulator(model)
estimator = do_mpc.estimator.StateFeedback(model)



"""
Set initial state
"""
np.random.seed(99)

#e = #np.ones([model.n_x,1])
# x0 = np.array([[0], [0.001],[0], [2.4037], [5.3453]]) #robot 206 
x0 = np.array([[0], [0.001],[0.0], [0], [0],[0], [0]]) # x0 = np.array([[0], [0.001],[0], [0], [12]]) #204
# x0 = np.array([[0], [0.001],[3.14], [12.8], [3.9]]) #robot 294 #np.random.uniform(-3*e,3*e) # Values between +3 and +3 for all states
mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

# Use initial state to set the initial guess.
mpc.set_initial_guess()

"""
Setup graphic:
"""

fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data)
plt.ion()

# wait for digit to be running
while True:
    try:
        ros.rosgraph.Master("/digit_main_node").getPid()
        break
    except socket.error: # raises connection refused if node doesn't exist yet
        pass

"""
Setup ROS publishing of the MPC state
"""
ros_publisher = rospy.Publisher("mpc", PredictionState)
rospy.init_node("mpc_publisher", anonymous=True)
ratelimiter = rospy.Rate(1/0.4) # time step 0.4 s

global feedback_valid
feedback_valid = False
global data_requested
data_requested = True

terminated = False
feedback_data = np.zeros(5)

global last_x, last_y, last_valid
last_x = 0
last_y = 0
last_valid = False


# mpc_data = PredictionState()
# mpc_data.x_com = x0[0]
# mpc_data.xd_com = x0[1]
# mpc_data.dtheta = x0[2]
# mpc_data.x = x0[3]
# mpc_data.y = x0[4]

# mpc_data.ux = 0
# mpc_data.uy = 0

# mpc_data.heading = x0[2]
# ros_publisher.publish(mpc_data)
print('MPC main loop')
"""
Run MPC main loop:
"""
t0 = time.time()

goal_x = 5
goal_y = 3

def feedback_callback(msg):
    start_time = time.time()
    global last_x, last_y, last_valid, x0, feedback_valid, data_requested
   
    # UNLISTIFY
    

    x0[0] = msg.x_com[0]
    x0[1] = msg.xd_com[0]
    x0[2] = msg.dtheta[0] 
    x0[3] = msg.x[0]
    x0[4] = msg.y[0]

    if np.sqrt((goal_x-msg.x[0])**2 + (goal_y-msg.y[0])**2)<0.5:
        mpc_data = PredictionState()
        mpc_data.x_com = mpc.data.prediction(('_x', 'x_com',0))[0][1:] #y_next[0]
        mpc_data.xd_com = 0.0
        mpc_data.dtheta = 0.0
        mpc_data.x = msg.x[0]
        mpc_data.y = msg.y[0]
        mpc_data.ux = 0.0
        mpc_data.uy = 0.0
        mpc_data.heading = 0.0

    else:
        mpc.x0 = x0
        simulator.x0 = x0
        estimator.x0 = x0
        
        
        
        u0 = mpc.make_step(x0) 
        y_next = simulator.make_step(u0) 
        x0 = estimator.make_step(y_next)
    

        delta_xk = y_next[0] - last_x
        delta_yk = y_next[2] - last_y
        heading = atan2(delta_yk,delta_xk)
        mpc_data = PredictionState()
        mpc_data.x_com = mpc.data.prediction(('_x', 'x_com',0))[0][1:] #y_next[0]
        mpc_data.xd_com = mpc.data.prediction(('_x', 'xd_com',0))[0][1:]
        mpc_data.dtheta = mpc.data.prediction(('_u', 'u_d_theta',0))[0][1:]
        mpc_data.x = mpc.data.prediction(('_x', 'x_g',0))[0][1:]
        mpc_data.y = mpc.data.prediction(('_x', 'y_g',0))[0][1:]
        mpc_data.ux = u0[0]
        mpc_data.uy = u0[1]
        mpc_data.heading = heading
        
    ros_publisher.publish(mpc_data)
    

    last_x = y_next[0]
    last_y = y_next[2]

    
feedback_subscriber = rospy.Subscriber("/mpc_feedback", PredictionState, feedback_callback, queue_size=1)

rospy.spin()



input('Press any key to exit.')

# Store results:
if store_results:
    do_mpc.data.save_results([mpc, simulator], 'digit')
