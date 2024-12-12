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
sys.path.append("../../social_nav/datatext/")
sys.path.append("../../social_nav/saved_models/")
sys.path.append("../../social_nav/scripts/")
sys.path.append("../../social_nav/utils/")
# sys.path.append("scripts/")

import do_mpc
import pickle as pkl

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

""" User settings: """
show_animation = False
store_results = True
save_freq = True
freq_tosave = []

"""
Get configured do-mpc modules:
"""
for trial in range(1):

    model = template_model()
    mpc = template_mpc(model)
    simulator = template_simulator(model)
    estimator = do_mpc.estimator.StateFeedback(model)

    if show_animation:
        fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data)
        plt.ion()

    """
    Set initial state
    """

    np.random.seed(99)

    #e = #np.ones([model.n_x,1])
    x_r_0 = 2.691638231277466
    y_r_0 = 5.364117622375488
    x0 = np.array([[0], [0.001],[0], [0], [0], [0], [0]]) #204 #np.random.uniform(-3*e,3*e) # Values between +3 and +3 for all states
    # x0[4] = np.random.default_rng().uniform(0, 13, size=(1, 1))
    x0[-1] = x0[4]
    mpc.x0 = x0
    simulator.x0 = x0
    estimator.x0 = x0

    # Use initial state to set the initial guess.
    mpc.set_initial_guess()

    """
    Setup graphic:
    """

    # fig, ax, graphics = do_mpc.graphics.default_plot(mpc.data)
    # plt.ion()


    """
    Run MPC main loop:
    """
    freq = []
    tvp_template = mpc.get_tvp_template()
    for k in range(100):

        t = time.time()
        u0 = mpc.make_step(x0)
        y_next = simulator.make_step(u0)
        x0 = estimator.make_step(y_next)

        x0[5]= 5
        x0[6] = 5
        
        elapsed = time.time() - t
        freq.append(1/elapsed)
        # print(elapsed)
        # print('freq: ', 1/elapsed)

        if show_animation:
            graphics.plot_results(t_ind=k)
            graphics.plot_predictions(t_ind=k)
            graphics.reset_axes()
            plt.show()
            plt.pause(0.01)

    print(np.mean(freq))
    freq_tosave.append(freq)
    # input('Press any key to exit.')

    # Store results:
    if store_results:
        savename = "standing"
        do_mpc.data.save_results([mpc, simulator], savename)

if save_freq:
    savename = "standing"
    if (not os.path.exists("results/freq")):
        os.mkdir(os.path.join(os.getcwd(), "results/freq"))
    path = os.path.join(os.getcwd(), "results/freq/" + savename + "_freq.pkl")
    freq_tosave = np.asarray(freq_tosave).T
    with open(path, 'wb') as f:
        pkl.dump(freq_tosave, f)