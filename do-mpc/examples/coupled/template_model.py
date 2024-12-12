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
   # model.set_expression(expr_name='cost', expr=sum1(_x**2))
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
   

   

   ped_past =  model.set_variable('_tvp', 'ped_past', shape=(ped_number,16))
   ped_init =  model.set_variable('_tvp', 'ped_init', shape=(ped_number,2))
   z_tvp = model.set_variable('_tvp', 'z_tvp', shape=(16,1))



   goal_G = model.set_variable('_tvp', 'goal_G', shape=(2,1))

   ped_Gx1 = model.set_variable('_tvp', 'ped_Gx1', shape=(8,1))
   ped_Gy1 = model.set_variable('_tvp', 'ped_Gy1', shape=(8,1))


   ped_num = model.set_variable('_tvp', 'ped_num')
   ped_num_adj =  model.set_variable('_tvp', 'ped_num_adj', shape=(ped_number,16))
   ped_num_flag =  model.set_variable('_tvp', 'ped_num_flag', shape=(ped_number,1))

   vx_t = model.set_variable('_tvp', 'vx_t')
   xd_com = model.set_variable('_x', 'xd_com')
   #y_com = model.set_variable('_x', 'y_com')
   #yd_com = model.set_variable('_x', 'yd_com')
   d_theta = model.set_variable('_x', 'd_theta')

   x_g = model.set_variable('_x', 'x_g')
   y_g = model.set_variable('_x', 'y_g')

   x_g_digit = model.set_variable('_x', 'x_g_digit')
   y_g_digit = model.set_variable('_x', 'y_g_digit')
   # s = model.set_variable('_u', 's')

   pf_x = model.set_variable('_u',  'pf_x')
   #pf_y = model.set_variable('_u',  'pf_y')
   ud_theta = model.set_variable('_u', 'u_d_theta')

   w = np.sqrt(9.81/1.02)
   T = 0.4


   x_n =  (pf_x-pf_x*np.cosh(w*T) + xd_com*np.sinh(w*T)/w)
   xd_n = -pf_x*w*np.sinh(w*T) + xd_com*np.cosh(w*T)

   #y_n = -pf_y*np.cosh(w*T) + yd_com*np.sinh(w*T)/w
   #yd_n = -pf_y*w*np.sinh(w*T) + yd_com*np.cosh(w*T)

   d_theta_n = d_theta + ud_theta

   x_g_n = x_g + (x_n*np.cos(d_theta_n)) #+ y_n*np.sin(d_theta) )

   y_g_n = y_g + (x_n*np.sin(d_theta_n)) #+ y_n*np.cos(d_theta) )


   model.set_rhs('x_com', x_n)
   model.set_rhs('xd_com', xd_n)
   #model.set_rhs('y_com', y_n)
   #model.set_rhs('yd_com', yd_n)

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
   # model.set_rhs('xg',0, x_g_n)
   # model.set_rhs('xg',1, y_g_n)
   ################################ onnx ################################
   
   # onnx_model = onnx.load("zono8_onnx_zono_curr_nxt_10ade_fde_reduce_size_03_005_stl.onnx")

   # old_input_name = onnx_model.graph.input[0].name
   # new_input_name = 'first_input'
   # onnx_model.graph.input[0].name = new_input_name
   # old_input_name = onnx_model.graph.input[1].name
   # new_input_name = 'second_input'
   # onnx_model.graph.input[1].name = new_input_name
   # old_input_name = onnx_model.graph.input[2].name
   # new_input_name = 'third_input'
   # onnx_model.graph.input[2].name = new_input_name
   # graph = onnx_model.graph

   # graph.node[0].input[0] = 'first_input'
   # graph.node[6].input[0] = 'second_input'
   # graph.node[11].input[2] = 'third_input'


   

   # onnx_model = onnx.load("zono13_batch_relu_reduce_size_nnextstate.onnx")
   ego_model = onnx.load("zono14_future_highpn_mean0201_stl.onnx")
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
   

   # ped_model = onnx.load("ped_pedoptimal3_coupled_highpn_global_sum_size.onnx")
   # ped_model = onnx.load("ped_pedoptimal4_coupled_highpn_global_sum_size.onnx")
   ped_model = onnx.load("ped_pedoptimal5_coupled_highpn_global_sum_size.onnx")
   # ped_model = onnx.load("ped_pedoptimal6_coupled_highpn_global_sum_size.onnx")

   graph_ped = ped_model.graph
   i = 0
   # print(ped_model.graph.input)
   # for node in graph_ped.node:
   #    print(i)
   #    i = i + 1
   #    print(f"Node Name: {node.name}")
   #    print(f"Operation Type: {node.op_type}")
   #    print(f"Input Names: {node.input}")
   #    print(f"Output Names: {node.output}")
   #    print("-----")

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

   ##### pedoptimal4
   # graph_ped.node[0].input[0] = 'past_input'
   # graph_ped.node[5].input[0] = 'ego_input'
   # graph_ped.node[10].input[-1] = 'z_input'
   # graph_ped.node[23].input[-1] = 'init_pos'
   
   ##### pedoptimal5
   graph_ped.node[0].input[0] = 'past_input'
   graph_ped.node[3].input[0] = 'ego_input'
   graph_ped.node[6].input[-1] = 'z_input'
   graph_ped.node[17].input[-1] = 'init_pos'
   
   ##### pedoptimal6
   # graph_ped.node[0].input[0] = 'past_input'
   # graph_ped.node[3].input[0] = 'ego_input'
   # graph_ped.node[6].input[-1] = 'z_input'
   # graph_ped.node[13].input[-1] = 'init_pos'
   


   graph_ped.node[-1].output[0] = 'output'

   casadi_ped_model = do_mpc.sysid.ONNXConversion(ped_model)

   # print(ped_model.graph.input)
   # for node in graph_ped.node:
   #    print(i)
   #    i = i + 1
   #    print(f"Node Name: {node.name}")
   #    print(f"Operation Type: {node.op_type}")
   #    print(f"Input Names: {node.input}")
   #    print(f"Output Names: {node.output}")
   #    print("-----")

   # ped_Gr1 = SX.sym('ped_Gr', 16, 1) 
   # for i in range(8):
   #    ped_Gr1[i * 2,:] = ped_Gx1[i,:] - x_g*model.tvp['ped_num']
   #    ped_Gr1[i * 2+1,:] = ped_Gy1[i,:] - y_g*model.tvp['ped_num']
   
   
   # print(ped_Gr.T.shape)

   # I = DM.eye(model.tvp['ped_num'])
   # print(I)
   # ped_size = []
   # ped_size.extend([ped_Gr.T[:model.tvp['ped_num'],:]])
   # model.set_expression('ped_size', ped_Gr.T[:model.tvp['ped_num'],:])

   z = model.tvp['z_tvp'].T
   past_input = model.tvp['ped_past'] #SX.sym('p', ped_number,16)
   init_pos = model.tvp['ped_init'] #SX.sym('init', ped_number+1,2) 
   casadi_ped_model.convert(past_input=past_input, init_pos=init_pos, ego_input=repmat(horzcat(*[x_n*np.cos(d_theta), x_n*np.sin(d_theta)]), ped_number), z_input=repmat(z, ped_number), verbose=False)
   ped_output = casadi_ped_model['output']
   ped_centers = SX.zeros(1 ,2) 
   peds_centers = SX.zeros(1,16)

   for i in range(ped_number):
      one_ped = reshape(ped_output[i,:], (10,7))
      exist = model.tvp['ped_num_flag'][i]
      for k in range(7):
         center = (one_ped.T[k,0:2] + horzcat(*[x_g, y_g]))*exist
         ped_centers = horzcat(ped_centers,center)
      peds_centers = vertcat(peds_centers,ped_centers)
      ped_centers = SX.zeros(1 ,2) 
   
   # print(peds_centers.shape)
   future_sum = sum1(peds_centers[1:,:]) #+ model.tvp['ped_num_adj'].T
   # future_sum = ped_Gr1.T
      
   # input1 = ped_Gr1.T #ped_Gr.T ############# pedestrians relative position 
   input2 = model.tvp['goal_G'].T - horzcat(*[x_g, y_g]) #casadi.SX.sym("in2",1,2) ############# goal relative position 
   
   
   casadi_ego_model.convert(first_input=future_sum, second_input=input2, third_input=horzcat(*[x_n*np.cos(d_theta), x_n*np.sin(d_theta)]), fourth_input=z, verbose=False)
   
   # print(ped_output.shape)
   # ped_NN_output = reshape(ped_output, (ped_number, 10, 7))
   # print(ped_NN_output.shape)
   # ped_centers =  ped_NN_output.T[:,:,0:2]
   # print(ped_centers.shape)
   
   ### Mask
   ### need to mask non observed pedestrians
   
   output = casadi_ego_model['output']
   NN_output = reshape(output, (10,7))
   # NN_output = reshape(casadi_converter['52'], (7,10))
   

   center = NN_output.T[0,0:2] + horzcat(*[x_g, y_g]) #(horzcat(*[x_g_n, y_g_n]) + horzcat(*[x_g, y_g]))*0.5
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
      G_ped = reshape(ped_output[ped,:], (10,7))
     
      # Mink_G = horzcat(*[G,model.tvp[f'ped_G{ped}1'], model.tvp[f'ped_G{ped}2'], model.tvp[f'ped_G{ped}3'], model.tvp[f'ped_G{ped}4']])
      
      Mink_G = horzcat(*[G,G_ped[2:4,0],G_ped[4:6,0], G_ped[6:8,0], G_ped[8:,0]])

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

      Mink_b_polyn = (Mink_A_poly @ centern.T) + fabs(Mink_ATG) @ Mink_ones

      ped_pos = G_ped[:2,0] + center.T

      model.set_expression(f'obs_zono_conts{ped}', -mmax(Mink_A_poly@model.tvp[f'obs_loc{ped}'] - Mink_b_poly))
      model.set_expression(f'nn_obs_loc{ped}',ped_pos)
      model.set_expression(f'G{ped}',horzcat(*[G_ped[2:4,0],G_ped[4:6,0], G_ped[6:8,0], G_ped[8:,0]]))
      # model.set_expression(f'obs_zono_conts{ped}', -mmax(Mink_A_poly@ped_pos - Mink_b_poly))
      # model.set_expression(f'obs_zono_conts{ped}',((1-gamma)*mmax(Mink_A_poly@model.tvp[f'obs_loc{ped}'] - Mink_b_poly) - mmax(Mink_A_poly@model.tvp[f'obs_loc{ped}n'] - Mink_b_polyn)))


   
   
   ######################## Solve for vertices for plotting################################

   model.set_expression('NN', output.T)
   model.set_expression('NN_ped', ped_output)
   model.set_expression('peds_centers', peds_centers)
   ###################### Constraints #######################################################

   zonotope_const = []


   model.set_expression('zonotope_const', mmax(A_poly@vertcat(*[x_g_n, y_g_n]) - b_poly))



   obs_zono_conts = []
   

   model.setup()


   return model
