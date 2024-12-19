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
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc
import torch
import torch.nn as nn
import argparse
sys.path.append("../../social_nav/datatext/")
sys.path.append("../../social_nav/saved_models/")
sys.path.append("../../social_nav/scripts/")
sys.path.append("../../social_nav/utils/")
from pred_egonet import*
from pred_zononet import*
from pkl_gen import *
import itertools


import onnx


def template_model(symvar_type='SX'):
   """
   --------------------------------------------------------------------------
   template_model: Variables / RHS / AUX
   --------------------------------------------------------------------------
   """
   model_type = 'discrete' # either 'discrete' or 'continuous'
   model = do_mpc.model.Model(model_type, symvar_type)

   # Simple oscillating masses example with two masses and two inputs.
   # States are the position and velocitiy of the two masses.

   # States struct (optimization variables):
   _xg = model.set_variable('_x', 'xg', shape=(2,1))

   # Input struct (optimization variables):
   _u = model.set_variable(var_type='_u', var_name='u', shape=(2,1))

   
   # Set expression. These can be used in the cost function, as non-linear constraints
   # or just to monitor another output.
   model_type = 'discrete' # either 'discrete' or 'continuous'
   model = do_mpc.model.Model(model_type, symvar_type)

   n_g = 7
   stance = model.set_variable('_tvp', 'stance')
   theta_g = model.set_variable('_tvp', 'theta_g')
   theta_t = model.set_variable('_tvp', 'theta_t')
   obsx= model.set_variable('_tvp', 'obsx')
   obsy= model.set_variable('_tvp', 'obsy')
   obsxn= model.set_variable('_tvp', 'obsxn')
   obsyn= model.set_variable('_tvp', 'obsyn')
   xg_running = model.set_variable('_tvp', 'xg_running')
   yg_running = model.set_variable('_tvp', 'yg_running')
   xg = model.set_variable('_tvp', 'xg')
   yg = model.set_variable('_tvp', 'yg')
   x_com = model.set_variable('_x', 'x_com')
   vx_running = model.set_variable('_tvp', 'vx_running')

   horz_slack = model.set_variable('_tvp', 'horz_slack')
   # ped_fut_sum = model.set_variable('_tvp', 'ped_fut_sum',(1,16))
   A = model.set_variable('_tvp', 'A', shape=(2*n_g,2))
   b = model.set_variable('_tvp', 'b', shape=(2*n_g,1))

   ATG = model.set_variable('_tvp', 'ATG', shape=(2*n_g,1))

   

   A_obs = model.set_variable('_tvp', 'A_obs', shape=(2*n_g,2))
   b_obs = model.set_variable('_tvp', 'b_obs', shape=(2*n_g,1))

   vertices = model.set_variable('_tvp', 'vertices', shape=(256,5))

   max_obs = model.set_variable('_tvp', 'max_obs', shape=(15*6,1))

   ped_G = model.set_variable('_tvp', 'ped_G', shape=(16,2))
   
   ped_number = 5
   for i in range(ped_number):
      model.set_variable('_tvp', f'obs_loc{i}', shape=(2,1))
      model.set_variable('_tvp', f'obs_loc{i}n', shape=(2,1))
      model.set_variable('_tvp', f'obs_loc{i}g', shape=(2,1))
      model.set_variable('_tvp', f'obs_loc{i}ng', shape=(2,1))
      model.set_variable('_tvp', f'ped_G{i}1', shape=(2,1))
      model.set_variable('_tvp', f'ped_G{i}2', shape=(2,1))
      model.set_variable('_tvp', f'ped_G{i}3', shape=(2,1))
      model.set_variable('_tvp', f'ped_G{i}4', shape=(2,1))


   model.set_variable('_tvp', 'GP_G', shape=(2,2))
   

   

   ped_past =  model.set_variable('_tvp', 'ped_past', shape=(ped_number+1,16))
   ped_init =  model.set_variable('_tvp', 'ped_init', shape=(ped_number+1,2))
   z_tvp = model.set_variable('_tvp', 'z_tvp', shape=(16,1))



   goal_G = model.set_variable('_tvp', 'goal_G', shape=(2,1))

   ped_Gx1 = model.set_variable('_tvp', 'ped_Gx1', shape=(8,1))
   ped_Gy1 = model.set_variable('_tvp', 'ped_Gy1', shape=(8,1))


   ped_num = model.set_variable('_tvp', 'ped_num')


   vx_t = model.set_variable('_tvp', 'vx_t')
   xd_com = model.set_variable('_x', 'xd_com')
   
   d_theta = model.set_variable('_x', 'd_theta')

   x_g = model.set_variable('_x', 'x_g')
   y_g = model.set_variable('_x', 'y_g')

   x_g_digit = model.set_variable('_x', 'x_g_digit')
   y_g_digit = model.set_variable('_x', 'y_g_digit')

   pf_x = model.set_variable('_u',  'pf_x')
   ud_theta = model.set_variable('_u', 'u_d_theta')

   w = np.sqrt(9.81/1.02)
   T = 0.4

   d_theta_n = d_theta + ud_theta
   x_n =  (pf_x-pf_x*np.cosh(w*T) + xd_com*np.sinh(w*T)/w)
   xd_n = -pf_x*w*np.sinh(w*T)*np.cos(d_theta_n) + xd_com*np.cosh(w*T)

   

   x_g_n = x_g + (x_n*np.cos(d_theta_n)) 
   y_g_n = y_g + (x_n*np.sin(d_theta_n)) 


   model.set_rhs('x_com', x_n)
   model.set_rhs('xd_com', xd_n)


   model.set_rhs('d_theta', d_theta_n)
   model.set_rhs('x_g', x_g_n)
   model.set_rhs('y_g', y_g_n)

   model.set_rhs('x_g_digit',x_g_digit)
   model.set_rhs('y_g_digit',y_g_digit)


   radius = 0.3

   gamma = 0.5

   obstacle_distance = []

   _xg[0] = x_g_n
   _xg[1] = y_g_n
   
   ################################ onnx ################################
      

   ego_model = onnx.load("zono14_future_highpn_mean0201_wostl.onnx")
   # ego_model = onnx.load("zono14_future_highpn_mean0201_stl.onnx") 

   graph = ego_model.graph

   # for node in graph.node:
   #    print(f"Node Name: {node.name}")
   #    print(f"Operation Type: {node.op_type}")
   #    print(f"Input Names: {node.input}")
   #    print(f"Output Names: {node.output}")
   #    print("-----")

   old_input_name = ego_model.graph.input[0].name
   new_input_name = 'first_input'
   ego_model.graph.input[0].name = new_input_name
   old_input_name = ego_model.graph.input[1].name
   new_input_name = 'second_input'
   ego_model.graph.input[1].name = new_input_name
   old_input_name = ego_model.graph.input[2].name
   new_input_name = 'third_input'
   ego_model.graph.input[2].name = new_input_name
   old_input_name = ego_model.graph.input[3].name
   new_input_name = 'fourth_input'
   ego_model.graph.input[3].name = new_input_name
   graph = ego_model.graph

   graph.node[0].input[0] = 'first_input'
   graph.node[5].input[0] = 'second_input'
   graph.node[10].input[0] = 'third_input'
   graph.node[15].input[-1] = 'fourth_input'

   graph.node[-1].output[0] = 'output'

   casadi_ego_model = do_mpc.sysid.ONNXConversion(ego_model)
   

   ped_model = onnx.load("ped_pedoptimal4_coupled_highpn_global_sum_size.onnx")
   graph_ped = ped_model.graph
   i = 0
  

   old_input_name = ped_model.graph.input[0].name
   new_input_name = 'past_input'
   ped_model.graph.input[0].name = new_input_name
   old_input_name = ped_model.graph.input[1].name
   new_input_name = 'init_pos'
   ped_model.graph.input[1].name = new_input_name
   old_input_name = ped_model.graph.input[2].name
   new_input_name = 'ego_input'
   ped_model.graph.input[2].name = new_input_name
   old_input_name = ped_model.graph.input[3].name
   new_input_name = 'z_input'
   ped_model.graph.input[3].name = new_input_name
   graph_ped = ped_model.graph

   graph_ped.node[0].input[0] = 'past_input'
   graph_ped.node[5].input[0] = 'ego_input'
   graph_ped.node[10].input[-1] = 'z_input'
   graph_ped.node[23].input[-1] = 'init_pos'

   graph_ped.node[-1].output[0] = 'output'

   casadi_ped_model = do_mpc.sysid.ONNXConversion(ped_model)

  

   ped_Gr1 = SX.sym('ped_Gr', 16, 1) 
   for i in range(8):
      ped_Gr1[i * 2,:] = ped_Gx1[i,:] - x_g*model.tvp['ped_num']
      ped_Gr1[i * 2+1,:] = ped_Gy1[i,:] - y_g*model.tvp['ped_num']
   
   
 

   z = model.tvp['z_tvp'].T
   
   future_sum = ped_Gr1.T
      
   input2 = model.tvp['goal_G'].T - horzcat(*[x_g, y_g]) 
   
   
   casadi_ego_model.convert(first_input=future_sum, second_input=input2, third_input=horzcat(*[x_n*np.cos(d_theta), x_n*np.sin(d_theta)]), fourth_input=z, verbose=False)
   
  
   ### Mask
   ### need to mask non observed pedestrians
   
   output = casadi_ego_model['output']
   NN_output = reshape(output, (10,7))
   

   center = NN_output.T[0,0:2] + horzcat(*[x_g, y_g]) 
   centern = NN_output.T[0,0:2] + horzcat(*[x_g_n, y_g_n])

 

   G_nn = reshape(NN_output.T[0,2:], (2,4))
   
   model.set_expression('center_out', center.T)
   
   
   G = horzcat(*[G_nn,[0.15,0], [0, 0.15], model.tvp['GP_G']])

   L = sqrt(G[0,:]*G[0,:]+G[1,:]*G[1,:])
   A_poly_in = SX.zeros(8, 2)
   for i in range(8):
      A_poly_in[i,0] =  1/L[i] * -G[1,i]
      A_poly_in[i,1] =  1/L[i] * G[1,i]

   A_poly = vertcat(*[A_poly_in,-A_poly_in])
   ATc = A_poly @ center.T
   ATG = A_poly_in @ G
   ATG = vertcat(*[ATG,-ATG])
   ones = SX.ones(8,1)
   b_poly = ATc + fabs(ATG) @ ones
   

  

   gamma = 0.2
   for ped in range(ped_number):
      Mink_G = horzcat(*[G,model.tvp[f'ped_G{ped}1'], model.tvp[f'ped_G{ped}2'], model.tvp[f'ped_G{ped}3'], model.tvp[f'ped_G{ped}4']])
      Mink_L = sqrt(Mink_G[0,:]*Mink_G[0,:]+Mink_G[1,:]*Mink_G[1,:])
      Mink_A_poly_in = SX.zeros(12, 2)
      for i in range(12):
         Mink_A_poly_in[i,0] =  1/Mink_L[i] * -Mink_G[1,i]
         Mink_A_poly_in[i,1] =  1/Mink_L[i] * Mink_G[0,i]

      Mink_A_poly = vertcat(*[Mink_A_poly_in,-Mink_A_poly_in])
      Mink_ATc = Mink_A_poly @ center.T
      Mink_ATG= Mink_A_poly_in @ Mink_G
      Mink_ATG = vertcat(*[Mink_ATG,-Mink_ATG])
      Mink_ones = SX.ones(12,1)
      Mink_b_poly = Mink_ATc + fabs(Mink_ATG) @ Mink_ones

      

      

      model.set_expression(f'obs_zono_conts{ped}', -mmax(Mink_A_poly@model.tvp[f'obs_loc{ped}'] - Mink_b_poly))


   
   ######################## Solve for vertices for plotting################################

   model.set_expression('NN', output.T)
   ###################### Constraints #######################################################

   zonotope_const = []


   model.set_expression('zonotope_const', mmax(A_poly@vertcat(*[x_g_n, y_g_n]) - b_poly))



   obs_zono_conts = []
   

   model.setup()


   return model
