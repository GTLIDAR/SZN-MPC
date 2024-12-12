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
from numpy.random import default_rng
from casadi import *
from casadi.tools import *
import itertools
import sys
sys.path.append('../../')
import do_mpc
import torch
import torch.nn as nn
import onnx




# sys.path.append("../utils/")
# sys.path.append("scripts/")

sys.path.append("../../social_nav/datatext/")
sys.path.append("../../social_nav/saved_models/")
sys.path.append("../../social_nav/scripts/")
sys.path.append("../../social_nav/utils/")

from pred_egonet import*
from pred_zononet import*
from pkl_gen import *

from GP import MPC_GP

from ped_trajectory import generate_ped_trajectory_tensor, generate_random_trajectories #sgan


"""
prediction module 
"""
parser = argparse.ArgumentParser(description='PECNet')

parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--load_file', '-lf', default="ped_zono_pred_size_sum_with_pn_mean0302.pt") #"ego8_turn_and_vel_limit_zara1.pt") # egonet_ego8_univ_sum_wostl_total_test_loss_minibatch.pt") # #"egonet_ego8_univ_sum_wostl_total_test_loss_minibatch.pt") #"comb_loco_stl_004.pt") #"egonnet_ego8_turn_angle_and_speed_stl.pt") #"") #
parser.add_argument('--num_trajectories', '-nt', default=20) #number of trajectories to sample
parser.add_argument('--verbose', '-v', action='store_true')
parser.add_argument('--root_path', '-rp', default="./")

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
print(device)


checkpoint = torch.load('../../social_nav/saved_models/{}'.format(args.load_file), map_location=device)
hyper_params = checkpoint["hyper_params"]

print(hyper_params)
checkpoint = torch.load('../../social_nav/saved_models/{}'.format(args.load_file), map_location=device)
hyper_params = checkpoint["hyper_params"]

batch_size = 20
req_data_hist = 8
req_future = 8
radius = 4
train = 0
# dataset = "Zara1_mpc"
dataset = "students003_mpc"

# orientation = 2.54

# model_pt = EgoNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params["fdim"], hyper_params["zdim"], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
# model_pt = EgoNet_sum(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params["pdim"], hyper_params["ddim"], hyper_params["fdim"], hyper_params["zdim"], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
# model_pt = ZonoNet_sum(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params["pdim"], hyper_params["ddim"], hyper_params["fdim"], hyper_params["zdim"], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], hyper_params["Ng"], args.verbose)
model_pt = PECNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params["pdim"], hyper_params["ddim"], hyper_params["fdim"], hyper_params["zdim"], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], hyper_params["Ng"], args.verbose)
## get human data
# test_past_list, test_goal_list, test_waypoint_list, test_past_list_g, test_goal_list_g, test_waypoint_list_g = get_seq_data('/home/ashamsah3/human_prediction/do-mpc/social_nav/datatext/students003_val.txt', 1, hyper_params["past_length"], hyper_params["future_length"])
test_past_list, test_goal_list, test_waypoint_list, test_future_list, test_past_list_g, test_goal_list_g, test_waypoint_list_g, test_future_list_g  = read_from_pkl(dataset, batch_size, req_data_hist, req_future, radius, 0, dataset)
start = 1
# waypoint_x_0, waypoint_y_0 = pred_ego(3, model_pt, device, checkpoint, hyper_params, start, test_past_list, test_goal_list, test_waypoint_list, test_future_list)
# waypoint_x_0, waypoint_y_0 , A_poly, b_poly, generator_vectorsnp, generator_matrices, generator_vectors, centers, ATG = pred_zono(3, model_pt, device, checkpoint, hyper_params, start, test_past_list, test_goal_list, test_waypoint_list, test_future_list, 0, 0)
# print(A_poly[0,:,:].shape)

# A_detach = A_poly.detach()
# b_detach = b_poly.detach()
# A_detach = np.array(A_detach.cpu().numpy())
# b_detach = np.array(b_detach.cpu().numpy())

num_steps = print(len(test_past_list))

j = 1

################## Social Model


onnx_model = onnx.load("zono14_future_highpn_mean0201_wostl.onnx")
# onnx_model = onnx.load("zono14_future_highpn_mean0201_stl.onnx") 
graph = onnx_model.graph


old_input_name = onnx_model.graph.input[0].name
new_input_name = 'first_input'
onnx_model.graph.input[0].name = new_input_name
old_input_name = onnx_model.graph.input[1].name
new_input_name = 'second_input'
onnx_model.graph.input[1].name = new_input_name
old_input_name = onnx_model.graph.input[2].name
new_input_name = 'third_input'
onnx_model.graph.input[2].name = new_input_name
old_input_name = onnx_model.graph.input[3].name
new_input_name = 'fourth_input'
onnx_model.graph.input[3].name = new_input_name
graph = onnx_model.graph

graph.node[0].input[0] = 'first_input'
graph.node[5].input[0] = 'second_input'
graph.node[10].input[0] = 'third_input'
graph.node[15].input[-1] = 'fourth_input'

graph.node[-1].output[0] = 'output'



social_model = do_mpc.sysid.ONNXConversion(onnx_model)

ped_model = onnx.load("ped_pedoptimal3_coupled_highpn_global_sum_size.onnx")
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


def generate_linear_trajectory_tensor(n, time_steps, start, finish):
    # Initialize the trajectory tensor with zeros
    trajectory_tensor = np.zeros((n, time_steps, 2))

    for example_idx in range(n):
        # Generate a linear interpolation from start to finish for x and y coordinates
        x_interp = np.linspace(start[0], finish[0], time_steps)
        y_interp = np.linspace(start[1], finish[1], time_steps)

        # Store the interpolated coordinates in the tensor
        trajectory_tensor[example_idx, :, 0] = x_interp - example_idx*0.8
        trajectory_tensor[example_idx, :, 1] = y_interp + example_idx * 0.4
    return torch.from_numpy(trajectory_tensor)


# def generate_linear_trajectory_tensor(n, time_steps, start, finish):
#     # Initialize the trajectory tensor with zeros
#     trajectory_tensor = np.zeros((n, time_steps, 2))

#     for example_idx in range(n):
#         # Generate a linear interpolation from start to finish for x and y coordinates
#         if start[0] < finish[0]:
#             signx=1
#         else:
#             signx=-1
        
#         x_interp = np.arange(start[0], finish[0], signx* 0.15)
#         if x_interp.shape[0] >= trajectory_tensor.shape[1]:
#             trajectory_tensor[example_idx, :, 0] = x_interp[:time_steps] 
#         else:
#             trajectory_tensor[example_idx, :x_interp.shape[0], 0] = x_interp
#             trajectory_tensor[example_idx, x_interp.shape[0]:, 0] = x_interp[-1]
        
#         if start[1] < finish[1]:
#             signy=1
#         else:
#             signy=-1
        
#         y_interp = np.arange(start[1], finish[1], signy*0.15)
#         if y_interp.shape[0] >= trajectory_tensor.shape[1]:
#             trajectory_tensor[example_idx, :, 1] = y_interp[:time_steps] 
#         else:
#             trajectory_tensor[example_idx, :y_interp.shape[0], 1] = y_interp
#             trajectory_tensor[example_idx, y_interp.shape[0]:, 1] = y_interp[-1] 

        
#     return torch.from_numpy(trajectory_tensor)

def euclidean_distance(coord1, coord2):

    return np.sqrt(coord1**2 + coord2**2)

def remove_tensors_over_distance(tensors, r):
    # Extract the coordinates from the tensors
    coordinates = tensors[:, -1, :]

    # print(coordinates)

    # Create a mask to keep track of tensors within distance "r"
    keep_mask = np.ones(len(coordinates[:]), dtype=bool)

    # Check distances between each pair of tensors
    for i in range(len(coordinates)):
        # for j in range(i + 1, len(coordinates)):
        dist = euclidean_distance(coordinates[i,0], coordinates[i,1])
        if dist > r:
            # Mark both tensors for removal
            keep_mask[i] = False
            # keep_mask[j] = False

    # Keep only the tensors that are within distance "r"
    filtered_tensors = tensors[keep_mask]

    return filtered_tensors, keep_mask


# ped = 1 # Number of examples
# time_steps = 400  # Number of time steps
    
# st = np.random.randint(12, size=2)
# finish = np.random.randint(12, size=2)
# pedestrians = generate_linear_trajectory_tensor(ped, time_steps, st, finish)
# all_ped = pedestrians

# for num in range (ped_number):
#     st = np.random.uniform(low=-2.0, high=14, size=2) #(13, size=2)
#     finish = np.random.uniform(low=-2.0, high=14, size=2) #np.random.randint(13, size=2)
#     pedestrians = generate_linear_trajectory_tensor(ped, time_steps, st, finish)
#     all_ped = torch.cat((all_ped, pedestrians),dim=0)

ped = 1 # Number of examples
time_steps = 120  # Number of time steps
ped_number = 5 # Number of examples
    
st = np.random.randint(10, size=2)
finish = np.random.randint(10, size=2)
pedestrians = generate_linear_trajectory_tensor(ped, time_steps, st, finish)
all_ped = pedestrians

for num in range (ped_number):
    st = np.random.randint(13, size=2)
    finish = np.random.randint(13, size=2)
    pedestrians = generate_linear_trajectory_tensor(ped, time_steps, st, finish)
    all_ped = torch.cat((all_ped, pedestrians),dim=0)

# ------ GP -----------
all_csvFileName = [["2023-12-06-15-25-mpc_state.csv","2023-12-06-15-25-mpc_feedback.csv"],
                   ["2023-12-06-18-40-mpc_state.csv","2023-12-06-18-40-mpc_feedback.csv"],
                   ["2023-12-07-12-16-mpc_state.csv","2023-12-07-12-16-mpc_feedback.csv"]]

#Initialize and Train 2 GP models (one for errorX, other for errorY)
mpc_gp = MPC_GP(all_csvFileName)


def template_mpc(model):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    # ---------- initial trajectories sgan -----------
    # using the global variables
    global ped_traj
    global ped_disp
    global rob_traj
    global init

    # random
    ped_traj, ped_disp = generate_random_trajectories(ped_number, 8, -0.5, 13)
    ped_traj, ped_disp = generate_ped_trajectory_tensor(ped_traj, ped_disp, True)

    # linear 
    # ped_traj = all_ped[:,:8,...].permute(1,0,2).cuda()
    # ped_disp = (all_ped[:, 1:9,...] - all_ped[:, :8,...] ).permute(1,0,2).cuda()

    # intialise for both random and linear.  
    rob_traj = np.zeros((9,2))
    init = True
    #-----------------------------------------------------
    mpc = do_mpc.controller.MPC(model)
    
    suppress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0, 'ipopt.max_iter': 50}
    # suppress_ipopt = {'ipopt.print_level':3, 'ipopt.max_iter': 50}
    # suppress_ipopt = {'ipopt.max_iter': 50}


    setup_mpc = {
        'n_robust': 0,
        'n_horizon': 6,
        't_step': 1,
        'store_full_solution': True,
        #'nl_cons_check_colloc_points': True,
        'nlpsol_opts' : suppress_ipopt
    }

    mpc.set_param(**setup_mpc)
    mpc.set_param(store_full_solution=True)



    xdg=0
    ydg=0
    n_g = 4

   

    xg = 10
    yg = 10



    ### with social acceptibility 
    # mterm =   100*(model.x['xd_com'] - xdg)**2  + 3*((model.x['x_g'] - xg)**2 + (model.x['y_g'] - yg)**2) + (model.x['d_theta'] - model.tvp['theta_t'] )**2 
    # lterm = (model.x['xd_com'] - xdg)**2  + 1.5*((model.x['x_g'] -  model.tvp['xg_running'])**2 + (model.x['y_g'] - model.tvp['yg_running'])**2) + ((model.x['x_g'] - model.tvp['xg'])**2 + (model.x['y_g'] - model.tvp['yg'])**2) + (model.x['d_theta'] - model.tvp['theta_g'])**2 
    
    ### without social acceptibility 
    mterm = (xdg - model.x['xd_com'] )**2  + ((xg - model.x['x_g'])**2 + (yg - model.x['y_g'] )**2) + (model.x['d_theta'] - model.tvp['theta_t'] )**2 
    lterm = (xdg - model.x['xd_com'] )**2  + 3*((xg - model.x['x_g'])**2 + (yg - model.x['y_g'] )**2) + (model.x['d_theta'] - model.tvp['theta_t'])**2 

    
    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(pf_x = 100, u_d_theta = 10)
    # mpc.set_rterm(pf_x = 10, u_d_theta = 10)

    

    mpc.bounds['lower', '_x', 'x_com'] = -0.2
    mpc.bounds['lower', '_x', 'xd_com'] = -0.1
    #mpc.bounds['lower', '_x', 'y_com'] = -0.5
    #mpc.bounds['lower', '_x','yd_com'] = -1
    # mpc.bounds['lower', '_x', 'd_theta'] 
    # mpc.bounds['lower', '_u', 's'] = 0.8
    # mpc.bounds['upper', '_u', 's'] = 1.0

    mpc.bounds['upper', '_x', 'x_com'] = 0.2
    mpc.bounds['upper', '_x', 'xd_com'] = 1.0

    # mpc.bounds['lower','_u','u',2] =  -0.3
    # mpc.bounds['upper','_u','u',2] =  0.3
    mpc.bounds['lower','_u', 'pf_x'] = -0.1
    #mpc.bounds['lower','_u',  'pf_y'] = -0.6
    mpc.bounds['lower','_u','u_d_theta'] = -0.2617

    mpc.bounds['upper','_u', 'pf_x'] =0.4
    #mpc.bounds['upper','_u',  'pf_y'] = 0.6
    mpc.bounds['upper','_u','u_d_theta'] = 0.2617

   
   
    
    
    
   
    s=[]
    s.append(40)
    x_global=[]
    x_global.append(0.5)
    y_global=[]
    y_global.append(0.5)

    

    tvp_template = mpc.get_tvp_template()
    
   
    def tvp_fun(t_ind):

        ind = t_ind // setup_mpc['t_step']
        int_ind = int(ind)
        
        

        xcurr = mpc.data['_x','x_g']
        ycurr = mpc.data['_x','y_g']
        




        

        global j
        
        
        global waypoint_x_0
        global waypoint_y_0
        global orientation
        

        # tvp_template['_tvp',:, 'ped_Gx'] = np.zeros((8,1))*2
        # tvp_template['_tvp',:, 'ped_Gy'] = np.zeros((8,1))*2
        for i in range(ped_number):
            tvp_template['_tvp',:, f'obs_loc{i}'] = np.array([0, 0]) 
            tvp_template['_tvp',:, f'obs_loc{i}n'] = np.array([-10, -10])  
            tvp_template['_tvp',:, f'ped_G{i}1'] = numpy.array([0.002, 0.0001])
            tvp_template['_tvp',:, f'ped_G{i}2'] = numpy.array([0.0001, 0.002])
            tvp_template['_tvp',:, f'ped_G{i}3'] = numpy.array([0.002, 0.0001])
            tvp_template['_tvp',:, f'ped_G{i}4'] = numpy.array([0.0001, 0.002]) 

            # tvp_template['_tvp',:, 'ped_past'] = np.zeros((ped_number,16)).T
            # tvp_template['_tvp',:, 'ped_init'] = np.zeros((ped_number,2)).T
        # 
        tvp_template['_tvp',:, 'GP_G'] = numpy.array([[0.0001, 0.0],[0.0, 0.001]])

        # tvp_template['_tvp',:, 'ped_G11'] = numpy.array([0.002, 0.0001])
        # tvp_template['_tvp',:, 'ped_G12'] = numpy.array([0.0001, 0.002])
        # tvp_template['_tvp',:, 'ped_G13'] = numpy.array([0.002, 0.0001])
        # tvp_template['_tvp',:, 'ped_G14'] = numpy.array([0.0001, 0.002])

        # tvp_template['_tvp',:, 'ped_G21'] = numpy.array([0.002, 0.0001])
        # tvp_template['_tvp',:, 'ped_G22'] = numpy.array([0.0001, 0.002])
        # tvp_template['_tvp',:, 'ped_G23'] = numpy.array([0.002, 0.0001])
        # tvp_template['_tvp',:, 'ped_G24'] = numpy.array([0.0001, 0.002])

        # tvp_template['_tvp',:, 'ped_G31'] = numpy.array([0.002, 0.0001])
        # tvp_template['_tvp',:, 'ped_G32'] = numpy.array([0.0001, 0.002])
        # tvp_template['_tvp',:, 'ped_G33'] = numpy.array([0.002, 0.0001])
        # tvp_template['_tvp',:, 'ped_G34'] = numpy.array([0.0001, 0.002])

        # tvp_template['_tvp',:, 'ped_G41'] = numpy.array([0.002, 0.0001])
        # tvp_template['_tvp',:, 'ped_G42'] = numpy.array([0.0001, 0.002])
        # tvp_template['_tvp',:, 'ped_G43'] = numpy.array([0.002, 0.0001])
        # tvp_template['_tvp',:, 'ped_G44'] = numpy.array([0.0001, 0.002])

        # tvp_template['_tvp',:, 'ped_G51'] = numpy.array([0.002, 0.0001])
        # tvp_template['_tvp',:, 'ped_G52'] = numpy.array([0.0001, 0.002])
        # tvp_template['_tvp',:, 'ped_G53'] = numpy.array([0.002, 0.0001])
        # tvp_template['_tvp',:, 'ped_G54'] = numpy.array([0.0001, 0.002])

        # tvp_template['_tvp',:, 'ped_G61'] = numpy.array([0.002, 0.0001])
        # tvp_template['_tvp',:, 'ped_G62'] = numpy.array([0.0001, 0.002])
        # tvp_template['_tvp',:, 'ped_G63'] = numpy.array([0.002, 0.0001])
        # tvp_template['_tvp',:, 'ped_G64'] = numpy.array([0.0001, 0.002])

       
        

        # tvp_template['_tvp',:, 'obs_loc'] = np.array([0, 0]) 
        # tvp_template['_tvp',:, 'obs_loc2'] = np.array([0, 0]) 
        # tvp_template['_tvp',:, 'obs_loc3'] = np.array([0, 0]) 
        
        # tvp_template['_tvp',:, 'obs_loc4'] = np.array([0, 0]) 
        # tvp_template['_tvp',:, 'obs_loc5'] = np.array([0, 0]) 
        # tvp_template['_tvp',:, 'obs_loc6'] = np.array([0, 0]) 

        # tvp_template['_tvp',:, 'obs_locn'] = np.array([-10, -10]) 
        # tvp_template['_tvp',:, 'obs_loc2n'] = np.array([-10, -10]) 
        # tvp_template['_tvp',:, 'obs_loc3n'] = np.array([-10, -10]) 
        
        # tvp_template['_tvp',:, 'obs_loc4n'] = np.array([-10, -10]) 
        # tvp_template['_tvp',:, 'obs_loc5n'] = np.array([-10, -10]) 
        # tvp_template['_tvp',:, 'obs_loc6n'] = np.array([-10, -10]) 
        
        
        # start = 17
        
        # test_past_list[j] - new_torch 
        # waypoint_x, waypoint_y = pred_ego(3, model_pt, device, checkpoint, hyper_params, j, test_past_list, test_goal_list, test_waypoint_list)

        
        if int_ind > 2:
            
            xpred = mpc.data.prediction(('_x', 'x_g')) # curr to next 
            ypred = mpc.data.prediction(('_x', 'y_g')) # curr to next 

            xd_com = mpc.data.prediction(('_x', 'xd_com'))
            ud_theta= mpc.data.prediction(('_u', 'u_d_theta'))

            theta_t = np.arctan2(yg-ypred[0,0],xg-xpred[0,0]) 
           
            # j = j + 1

            tensor_robot_loc = torch.empty((1, 8, 2)) 
            ped_x = np.zeros((8,100))
            ped_y = np.zeros((8,100))

            ped_past_zero = np.zeros((ped_number+1, 16))
            ped_init_zero = np.zeros((ped_number+1, 2))
            social_ped_Gr1 = np.zeros((16,1))
            social_ped_x = np.zeros((8,100))
            social_ped_y = np.zeros((8,100))
            # print(model.aux['ped_size'].shape)
            tensor_robot_loc = torch.empty((1, 8, 2)) 
            xcurr_stack = np.ones(8) * xpred[0,0]
            ycurr_stack = np.ones(8) * ypred[0,0]
            
            tensor_robot_loc[0,:,:] = torch.FloatTensor(np.column_stack((xcurr_stack.T,ycurr_stack.T)))

            

                
            ######## custom pedestrians 
            # ped = 1 # Number of examples
            # time_steps = 100  # Number of time steps
            
            # st = np.array([8.0, 8.0])  # Start point (x, y)
            # finish = np.array([0.0, 0.0]) 
            # pedestrians = generate_linear_trajectory_tensor(ped, time_steps, st, finish)

            # st = np.array([14.0, 4.0])  # Start point (x, y)
            # finish = np.array([-14.0, 4.0])  # Finish point (x, y)
            # pedestrians2 = generate_linear_trajectory_tensor(ped, time_steps, st, finish)

            # st = np.array([3, 0.0])  # Start point (x, y)
            # finish = np.array([3, 16.0])  # Finish point (x, y)
            # pedestrians3 = generate_linear_trajectory_tensor(ped, time_steps, st, finish)

            # st = np.array([7.0, 12.0])  # Start point (x, y)
            # finish = np.array([7.0, -1.0]) # Finish point (x, y)
            # pedestrians4 = generate_linear_trajectory_tensor(ped, time_steps, st, finish)

            # st = np.array([0.0, 0.0])  # Start point (x, y)
            # finish = np.array([8.0, 8.0]) 
            # pedestrians5 = generate_linear_trajectory_tensor(ped, time_steps, st, finish)

            # st = np.array([7.0, 5.8])  # Start point (x, y)
            # finish = np.array([7.0, 5.8]) 
            # pedestrians6 = generate_linear_trajectory_tensor(ped, time_steps, st, finish)

            time_Step = 10+int_ind
            past_Step= 2+int_ind

            #---------------------------------Simulation using SGAN------------------------------------
            # using the global variables
            global ped_traj
            global ped_disp
            global rob_traj
            global init
            
            # history of robot
            if init:
                rob_traj = np.ones((9,2)) * np.asarray([xpred[0,0], ypred[0,0]]).T
                init = False
            else:
                rob_traj = np.append(rob_traj, np.asarray([xpred[0,0], ypred[0,0]]).T, axis=0)[1:]
            
            #convert robot history to tensor
            rob = torch.from_numpy(rob_traj[1:]).to(dtype= torch.double, device = "cuda").unsqueeze(1)
            rob_traj_disp = rob_traj[1:] - rob_traj[:-1]
            rob_disp = torch.from_numpy(rob_traj_disp).to(dtype= torch.double, device = "cuda").unsqueeze(1)

            #combine pedestrian and robot data
            traject = torch.concat([ped_traj, rob], dim=1)
            traject_disp = torch.concat([ped_disp, rob_disp], dim=1)

            #generate next step using sgan
            traject, traject_disp = generate_ped_trajectory_tensor(traject, traject_disp)

            #remove robot data
            
            ped_traj = traject[:,:-1,...]
            ped_disp = traject_disp[:,:-1,...]
            ## Borders
            ped_traj = torch.clamp(ped_traj, min=-1, max=13)
            min_indices = (ped_traj == -1).nonzero(as_tuple=True)
            max_indices = (ped_traj == 13).nonzero(as_tuple=True)
            ped_traj[max_indices] = ped_traj[max_indices] - 0.3
            ped_traj[min_indices] = ped_traj[min_indices] + 0.3
            ped_disp[max_indices] = -0.3
            ped_disp[min_indices] = 0.3
            #assign pedestrian data
            all_ped = ped_traj.permute(1, 0, 2).cpu()
            

            all_ped[3,:,:] = torch.ones_like(all_ped[3,:,:])*5.5
            all_ped[0,:,0] = torch.ones_like(all_ped[3,:,0])*5.5
            all_ped[0,:,1] = torch.ones_like(all_ped[3,:,1])*5.5
            all_ped[2,:,1] = torch.ones_like(all_ped[3,:,1])*5.0
            all_ped[2,:,0] = torch.ones_like(all_ped[3,:,0])*5
            all_ped[1,:,1] = torch.ones_like(all_ped[3,:,1])*5.5
            all_ped[1,:,0] = torch.ones_like(all_ped[3,:,0])*6
            all_ped[4,:,1] = torch.ones_like(all_ped[3,:,1])*6.5
            all_ped[4,:,0] = torch.ones_like(all_ped[3,:,0])*5.5
            # all_ped = torch.cat((pedestrians, pedestrians2, pedestrians3,pedestrians4,pedestrians5,pedestrians6),dim=0)
            

            test_past_list[start+j] =  all_ped - tensor_robot_loc #for simulation using Sgan
            # print[test_past_list[start+j].shape]
            filtered, keep_mask = remove_tensors_over_distance(test_past_list[start+j], 4)
            # test_past_list[start+j] = all_ped[:,past_Step:time_Step,:][keep_mask]
            for i in range(ped_number):
                tvp_template['_tvp',:, f'obs_loc{i}g'] = np.array([all_ped[i,-1,0],all_ped[i,-1,1]])
                tvp_template['_tvp',:, f'obs_loc{i}ng'] = np.array([all_ped[i,0,0],all_ped[i,0,1]])

            # -------------- Fully coupled 
            # tvp_template['_tvp',:, 'ped_past'] = (test_past_list[start+j].contiguous().view(-1, test_past_list[start+j].shape[1]*test_past_list[start+j].shape[2])).numpy()
            # tvp_template['_tvp',:, 'ped_init'] = (test_past_list[start+j].contiguous().view(-1, test_past_list[start+j].shape[1]*test_past_list[start+j].shape[2]))[:,:2].numpy()

            test_past_list[start+j] =  all_ped[keep_mask]

            


            


            # ------------- Fixed traj
            # test_past_list[start+j] =  all_ped[:,past_Step:time_Step,:] - tensor_robot_loc

            # for i in range(ped_number):
            #     tvp_template['_tvp',:, f'obs_loc{i}g'] = np.array([all_ped[i,time_Step,0],all_ped[i,time_Step,1]])
            #     tvp_template['_tvp',:, f'obs_loc{i}ng'] = np.array([all_ped[i,time_Step+1,0],all_ped[i,time_Step+1,1]])

            # filtered, keep_mask = remove_tensors_over_distance(test_past_list[start+j], 4)
            # # tvp_template['_tvp',:, 'ped_past'] = (filtered.contiguous().view(-1, filtered.shape[1]*filtered.shape[2])).numpy()
            # # tvp_template['_tvp',:, 'ped_init'] = (filtered.contiguous().view(-1, filtered.shape[1]*filtered.shape[2]))[:,-2:].numpy()
            
            # test_past_list[start+j] = all_ped[:,past_Step:time_Step,:][keep_mask]

            

            

            

           

            
            # test_past_list[start+j] =  (test_past_list_g[start+j] - tensor_robot_loc).flip(1)
            # # print(test_past_list[start+j].shape)
            # tvp_template['_tvp',:, 'ped_past'] = (test_past_list[start+j].contiguous().view(-1, test_past_list[start+j].shape[1]*test_past_list[start+j].shape[2])).numpy()
            # tvp_template['_tvp',:, 'ped_init'] = (test_past_list[start+j].contiguous().view(-1, test_past_list[start+j].shape[1]*test_past_list[start+j].shape[2]))[:,-2:].numpy()
            # test_future_list[start+j] = test_future_list_g[start+j] - tensor_robot_loc
            # filtered, keep_mask = remove_tensors_over_distance(test_past_list[start+j], 4)
            # # filtered_fut, _ = remove_tensors_over_distance(test_future_list[start+j], 4)
            # test_past_list[start+j] = test_past_list_g[start+j][:, :, :][keep_mask].flip(1)
            # test_future_list[start+j] = filtered_fut

            

            ped_num = test_past_list[start+j].numpy().shape[0]

            # print(ped_num)
            if ped_num > ped_number:
                ped_num = ped_number
            
            # print(ped_past_zero[:ped_num,:].shape)
            # print((test_past_list[start+j].contiguous().view(-1, test_past_list[start+j].shape[1]*test_past_list[start+j].shape[2])).numpy().shape)
            # ped_past_zero[:ped_num,:] = (test_past_list[start+j].contiguous().view(-1, test_past_list[start+j].shape[1]*test_past_list[start+j].shape[2])).numpy()
            # tvp_template['_tvp',:, 'ped_past'] = ped_past_zero #- ped_past_zero[:,:2]
            # tvp_template['_tvp',:, 'ped_init'] = ped_past_zero[:,-2:]

            tvp_template['_tvp',:, 'ped_num'] = ped_num

            ped_centers, ped_generator_vectors1 = ped_pred_zono(1, model_pt, device, checkpoint, hyper_params,  start+j, test_past_list, test_future_list)

            # print('##############################   ', ped_num)

            

            # xcurr_stack = np.ones(8) * xpred[0,n]
            # ycurr_stack = np.ones(8) * ypred[0,n]

                
            k = 0
            i = 0

            tvp_template['_tvp',:, 'goal_G'] = numpy.array([xg, yg])


            z = model.tvp['z_tvp'].T
   
            # casadi_converter.convert(first_input=input1, second_input=input2, third_input=horzcat(*[xd_n, d_theta_n]), fourth_input=z, verbose=False)
            
            
                

                
            ######## TXT pedestrians 
            
            # test_past_list[start+j] =  test_past_list_g[start+j] - tensor_robot_loc
            # test_future_list[start+j] = test_future_list_g[start+j] - tensor_robot_loc
            # filtered, keep_mask = remove_tensors_over_distance(test_past_list[start+j], 4)
            # filtered_fut, _ = remove_tensors_over_distance(test_future_list[start+j], 4)
            # test_past_list[start+j] = test_past_list_g[start+j][:, :, :][keep_mask]
            # test_future_list[start+j] = filtered_fut

            

            # ped_num = test_past_list[start+j].numpy().shape[0]
            # if ped_num > 6:
            #     ped_num = 6
            
            # tvp_template['_tvp',:, 'ped_num'] = ped_num

            # ped_centers, ped_generator_vectors1 = ped_pred_zono(1, model_pt, device, checkpoint, hyper_params,  start+j, test_past_list, test_future_list)

            # print('##############################   ', ped_num)

            if ped_num > 0:

                for n in range(setup_mpc['n_horizon']):   
                    X_gp = np.array([xd_com[0,n+1], ud_theta[0,n], xd_com[0,n]])
                    errorX_pred, errorX_cov, errorY_pred, errorY_cov = mpc_gp.predict_mpc_deviation(X_gp)
                    # tvp_template['_tvp',n+1, 'GP_G'] = np.array([[errorX_pred.item(), 0], [0,errorY_pred.item()]]) 
                    tvp_template['_tvp',:, 'GP_G'] = numpy.array([[0.0001, 0.0],[0.0, 0.001]])

                
                for k in range(ped_num):
                    ped_x[:,k] = test_past_list[start+j][k,:,0].numpy().T #- xcurr_stack
                    ped_y[:,k] = test_past_list[start+j][k,:,1].numpy().T #- ycurr_stack
                    
                    social_ped_x[:,k] = test_past_list[start+j][k,:,0].numpy().T - xcurr_stack
                    social_ped_y[:,k] = test_past_list[start+j][k,:,1].numpy().T - ycurr_stack

                sum_socialx = np.sum(social_ped_x, axis=1)
                sum_socialy = np.sum(social_ped_y, axis=1)
                for i in range(8):
                    social_ped_Gr1[i * 2,:] = sum_socialx[i] 
                    social_ped_Gr1[i * 2+1,:] = sum_socialy[i]

                past_input = (filtered.contiguous().view(-1, filtered.shape[1]*filtered.shape[2])).numpy() #SX.sym('p', ped_number,16)
                init_pos = (filtered.contiguous().view(-1, filtered.shape[1]*filtered.shape[2]))[:,-2:].numpy()#SX.sym('init', ped_number+1,2) 
                
                # print(past_input)
                # print(init_pos)
                # print(repmat(numpy.array([xpred[0,1]-xpred[0,0], ypred[0,1]-ypred[0,0]]).T, ped_num).shape)
                # print(ped_num)
                z = torch.empty(ped_num, 16)
                z.normal_(0, 1.3)
                z = z.numpy()
                # print(z.shape)
                casadi_ped_model.convert(past_input=past_input, init_pos=init_pos, ego_input=repmat(numpy.array([xpred[0,1]-xpred[0,0], ypred[0,1]-ypred[0,0]]).T, ped_num), z_input=z, verbose=False)
                ped_output = casadi_ped_model['output']
                ped_output = np.reshape(np.array(ped_output), (ped_num,7,10))
                ped_centers = ped_output[:,:,0:2]
                ped_generator_vectors1=np.reshape(ped_output[:,:,2:],(ped_num,7,4,2))


                pedx_fut = np.concatenate((test_past_list[start+j][:,-1,0].unsqueeze(-1).numpy(),ped_centers[:,:,0]),axis=1)
                pedy_fut = np.concatenate((test_past_list[start+j][:,-1,1].unsqueeze(-1).numpy(),ped_centers[:,:,1]),axis=1)

                sum_socialx = np.sum(pedx_fut, axis=0)
                sum_socialy = np.sum(pedy_fut, axis=0)
                for i in range(8):
                    social_ped_Gr1[i * 2,:] = sum_socialx[i] 
                    social_ped_Gr1[i * 2+1,:] = sum_socialy[i]
                    

            



                input1 = social_ped_Gr1.T #ped_Gr.T ############# pedestrians relative position 
                input2 = numpy.array([xg-xpred[0,0], yg-ypred[0,0]]).T
                z = torch.empty(1, 16)
                z.normal_(0, 1.3)
                z = z.numpy()

                # social_model.convert(first_input=input1, second_input=input2, third_input=0.1*numpy.array([xg-xpred[0,0], yg-ypred[0,0]]).T, fourth_input=z, verbose=False)
                social_model.convert(first_input=input1, second_input=input2, third_input=numpy.array([xpred[0,1]-xpred[0,0], ypred[0,1]-ypred[0,0]]).T, fourth_input=z, verbose=False)
                social_output = social_model['output']
                social_output = np.reshape(social_output, (7,10))

                social_waypoints = social_output[:,0:2] + numpy.array([xpred[0,0], ypred[0,0]]).T

                
                for n in range(setup_mpc['n_horizon']):
                    
                    for i in range(ped_num):
                        tvp_template['_tvp',n, f'obs_loc{i}'] = np.array([ped_centers[i,n,0]+xpred[0,0], ped_centers[i,n,1]+ypred[0,0]])
                        tvp_template['_tvp',n, f'obs_loc{i}n'] = np.array([ped_centers[i,n+1,0]+xpred[0,0], ped_centers[i,n+1,1]+ypred[0,0]])  
                        tvp_template['_tvp',n, f'ped_G{i}1'] = numpy.array([ped_generator_vectors1[i,n,0,0], ped_generator_vectors1[i,n,0,1]]) 
                        tvp_template['_tvp',n, f'ped_G{i}2'] = numpy.array([ped_generator_vectors1[i,n,1,0], ped_generator_vectors1[i,n,1,1]])
                        tvp_template['_tvp',n, f'ped_G{i}3'] = numpy.array([ped_generator_vectors1[i,n,2,0], ped_generator_vectors1[i,n,2,1]]) 
                        tvp_template['_tvp',n, f'ped_G{i}4'] = numpy.array([ped_generator_vectors1[i,n,3,0], ped_generator_vectors1[i,n,3,1]]) 

                    # if ped_num >= 1:
                    #     tvp_template['_tvp',n, 'obs_loc'] = np.array([ped_centers[0,n,0], ped_centers[0,n,1]])
                    #     tvp_template['_tvp',n, 'obs_locn'] = np.array([ped_centers[0,n+1,0], ped_centers[0,n+1,1]])  
                    #     tvp_template['_tvp',n, 'ped_G11'] = numpy.array([ped_generator_vectors1[0,n,0,0], ped_generator_vectors1[0,n,0,1]]) 
                    #     tvp_template['_tvp',n, 'ped_G12'] = numpy.array([ped_generator_vectors1[0,n,1,0], ped_generator_vectors1[0,n,1,1]])
                    #     tvp_template['_tvp',n, 'ped_G13'] = numpy.array([ped_generator_vectors1[0,n,2,0], ped_generator_vectors1[0,n,2,1]]) 
                    #     tvp_template['_tvp',n, 'ped_G14'] = numpy.array([ped_generator_vectors1[0,n,3,0], ped_generator_vectors1[0,n,3,1]]) 
                    #     if ped_num >= 2:
                    #         tvp_template['_tvp',n, 'obs_loc2'] = np.array([ped_centers[1,n,0], ped_centers[1,n,1]])
                    #         tvp_template['_tvp',n, 'obs_loc2n'] = np.array([ped_centers[1,n+1,0], ped_centers[1,n+1,1]])
                    #         tvp_template['_tvp',n, 'ped_G21'] = numpy.array([ped_generator_vectors1[1,n,0,0], ped_generator_vectors1[1,n,0,1]]) 
                    #         tvp_template['_tvp',n, 'ped_G22'] = numpy.array([ped_generator_vectors1[1,n,1,0], ped_generator_vectors1[1,n,1,1]])
                    #         tvp_template['_tvp',n, 'ped_G23'] = numpy.array([ped_generator_vectors1[1,n,2,0], ped_generator_vectors1[1,n,2,1]]) 
                    #         tvp_template['_tvp',n, 'ped_G24'] = numpy.array([ped_generator_vectors1[1,n,3,0], ped_generator_vectors1[1,n,3,1]]) 
                    #         if ped_num >= 3:
                    #             tvp_template['_tvp',n, 'obs_loc3'] = np.array([ped_centers[2,n,0], ped_centers[2,n,1]])
                    #             tvp_template['_tvp',n, 'obs_loc3n'] = np.array([ped_centers[2,n+1,0], ped_centers[2,n+1,1]])
                    #             tvp_template['_tvp',n, 'ped_G31'] = numpy.array([ped_generator_vectors1[2,n,0,0], ped_generator_vectors1[2,n,0,1]]) 
                    #             tvp_template['_tvp',n, 'ped_G32'] = numpy.array([ped_generator_vectors1[2,n,1,0], ped_generator_vectors1[2,n,1,1]])
                    #             tvp_template['_tvp',n, 'ped_G33'] = numpy.array([ped_generator_vectors1[2,n,2,0], ped_generator_vectors1[2,n,2,1]]) 
                    #             tvp_template['_tvp',n, 'ped_G34'] = numpy.array([ped_generator_vectors1[2,n,3,0], ped_generator_vectors1[2,n,3,1]]) 
                    #             if ped_num >= 4:
                    #                 tvp_template['_tvp',n, 'obs_loc4'] = np.array([ped_centers[3,n,0], ped_centers[3,n,1]])
                    #                 tvp_template['_tvp',n, 'obs_loc4n'] = np.array([ped_centers[3,n+1,0], ped_centers[3,n+1,1]])
                    #                 tvp_template['_tvp',n, 'ped_G41'] = numpy.array([ped_generator_vectors1[3,n,0,0], ped_generator_vectors1[3,n,0,1]]) 
                    #                 tvp_template['_tvp',n, 'ped_G42'] = numpy.array([ped_generator_vectors1[3,n,1,0], ped_generator_vectors1[3,n,1,1]])
                    #                 tvp_template['_tvp',n, 'ped_G43'] = numpy.array([ped_generator_vectors1[3,n,2,0], ped_generator_vectors1[3,n,2,1]]) 
                    #                 tvp_template['_tvp',n, 'ped_G44'] = numpy.array([ped_generator_vectors1[3,n,3,0], ped_generator_vectors1[3,n,3,1]])  
                    #                 if ped_num >= 5:
                    #                     tvp_template['_tvp',n, 'obs_loc5'] = np.array([ped_centers[4,n,0], ped_centers[4,n,1]])
                    #                     tvp_template['_tvp',n, 'obs_loc5n'] = np.array([ped_centers[4,n+1,0], ped_centers[4,n+1,1]])
                    #                     tvp_template['_tvp',n, 'ped_G51'] = numpy.array([ped_generator_vectors1[4,n,0,0], ped_generator_vectors1[4,n,0,1]]) 
                    #                     tvp_template['_tvp',n, 'ped_G52'] = numpy.array([ped_generator_vectors1[4,n,1,0], ped_generator_vectors1[4,n,1,1]])
                    #                     tvp_template['_tvp',n, 'ped_G53'] = numpy.array([ped_generator_vectors1[4,n,2,0], ped_generator_vectors1[4,n,2,1]]) 
                    #                     tvp_template['_tvp',n, 'ped_G54'] = numpy.array([ped_generator_vectors1[4,n,3,0], ped_generator_vectors1[4,n,3,1]]) 
                    #                     if ped_num >= 6:
                    #                         tvp_template['_tvp',n, 'obs_loc6'] = np.array([ped_centers[5,n,0], ped_centers[5,n,1]])
                    #                         tvp_template['_tvp',n, 'obs_loc6n'] = np.array([ped_centers[5,n+1,0], ped_centers[5,n+1,1]])
                    #                         tvp_template['_tvp',n, 'ped_G61'] = numpy.array([ped_generator_vectors1[5,n,0,0], ped_generator_vectors1[5,n,0,1]]) 
                    #                         tvp_template['_tvp',n, 'ped_G62'] = numpy.array([ped_generator_vectors1[5,n,1,0], ped_generator_vectors1[5,n,1,1]])
                    #                         tvp_template['_tvp',n, 'ped_G63'] = numpy.array([ped_generator_vectors1[5,n,2,0], ped_generator_vectors1[5,n,2,1]]) 
                    #                         tvp_template['_tvp',n, 'ped_G64'] = numpy.array([ped_generator_vectors1[5,n,3,0], ped_generator_vectors1[5,n,3,1]])  

                    # dist = ((ped_x)**2 + (ped_y)**2)**0.5
                    # print(dist)
                    

                    # tvp_template['_tvp',n, 'ped_G1'] = numpy.array([0.2, 0.01])
                    # tvp_template['_tvp',n, 'ped_G2'] = numpy.array([0.01, 0.2])

                    # tvp_template['_tvp',n, 'ped_Gx1'] = ped_x.T #np.sum(ped_x, axis=1).T
                    # tvp_template['_tvp',n, 'ped_Gy1'] = ped_y.T #np.sum(ped_y, axis=1).T
                    # print(test_past_list[start+j][:,-1,0])
                    # print(ped_centers[:,:,0].shape)
                    # print(pedx_fut)
                    # print(pedy_fut)

                    tvp_template['_tvp',:, 'ped_Gx1'] = np.sum(pedx_fut, axis=0).T #np.sum(np.flip(ped_x, axis=0), axis=1).T
                    tvp_template['_tvp',:, 'ped_Gy1'] = np.sum(pedy_fut, axis=0).T #np.sum(np.flip(ped_y, axis=0), axis=1).T
                    
                    tvp_template['_tvp',n, 'xg_running'] = social_waypoints[n+1,0]
                    tvp_template['_tvp',n, 'yg_running'] = social_waypoints[n+1,1]
                    tvp_template['_tvp',:, 'xg'] =  social_waypoints[-1,0]
                    tvp_template['_tvp',:, 'yg'] =  social_waypoints[-1,1]
                    theta_g = np.arctan2(social_waypoints[-1,1]-ypred[0,0],social_waypoints[-1,0]-xpred[0,0]) 
                    tvp_template['_tvp',n,'theta_g']  = (theta_g + np.pi) % (2 * np.pi) - np.pi

                    # tvp_template['_tvp',n, 'ped_G'] = test_past_list_g[startj+n][0:2,:].view(2,16).numpy().T
                    
                # tvp_template['_tvp',n, 'obs_loc'] = np.array([test_past_list_g[start+j+n][0,0,0], test_past_list_g[start+j+n][0,0,1]])
                
                # obsx = test_past_list_g[start+j+n][:,0,0].numpy() - xpred[0,0]
                # obsy = test_past_list_g[start+j+n][:,0,1].numpy() - ypred[0,0]
                # dist = ((obsx)**2 + (obsy)**2)**0.5
            
                # closest_obs_ind = np.argmin(dist)
                # tvp_template['_tvp',:, 'obs_loc'] = np.array([5, 5])

                # tvp_template['_tvp',n, 'obs_loc'] = np.array([8-0.1*(int_ind-1),5.0])#
                if np.sqrt((yg-ypred[0,0])**2+(xg-xpred[0,0])**2) <3:
                    tvp_template['_tvp',:, 'xg_running'] = xg
                    tvp_template['_tvp',:, 'yg_running'] = yg
                    tvp_template['_tvp',:, 'xg'] =  xg
                    tvp_template['_tvp',:, 'yg'] =  yg
                    tvp_template['_tvp',:,'theta_g']  = (theta_t + np.pi) % (2 * np.pi) - np.pi




            else:
                tvp_template['_tvp',:, 'xg_running'] = xg
                tvp_template['_tvp',:, 'yg_running'] = yg
                tvp_template['_tvp',:, 'xg'] =  xg
                tvp_template['_tvp',:, 'yg'] =  yg
                tvp_template['_tvp',:,'theta_g']  = (theta_t + np.pi) % (2 * np.pi) - np.pi


            
            tvp_template['_tvp',:,'theta_t'] = (theta_t + np.pi) % (2 * np.pi) - np.pi


            for n in range(setup_mpc['n_horizon']+1):
                tvp_template['_tvp',n,'horz_slack'] = n
                z = torch.empty(1, 16)
                z.normal_(0, 1.3)
                z = z.numpy()
                tvp_template['_tvp',n, 'z_tvp'] = z.T

            
                


            

                
                
        

            # print(mpc.opt_x_num['_x',:,0,0,'x_g'])
            # print(xpred)
            
            
            # for ob in range(3):
            #         tensor_robot_loc = torch.empty((1, 8, 2)) 
            #         xcurr_stack = np.ones(8) * xpred[0,0]
            #         ycurr_stack = np.ones(8) * ypred[0,0]
                    
                    
            #         tensor_robot_loc[0,:,:] = torch.FloatTensor(np.column_stack((xcurr_stack.T,ycurr_stack.T)))


            #         test_past_list[start+j] =  test_past_list_g[start+j] - tensor_robot_loc
            #         test_future_list[start+j] = test_future_list_g[start+j] - tensor_robot_loc

            #         filtered = remove_tensors_over_distance(test_past_list[start+j], 4)
            #         filtered_fut = remove_tensors_over_distance(test_future_list[start+j], 4)
                    
            #         test_past_list[start+j] = filtered
            #         test_future_list[start+j] = filtered_fut
                    
            #         if filtered.size(0) >= 1:
                    
            #             obsx = filtered_fut[:,0,0].numpy()
            #             obsy = filtered_fut[:,0,1].numpy()
            #             dist = (obsx**2 + obsy**2)**0.5
                        
            #             closest_obs_ind = np.argmin(dist)
            #             closest_obs_x = test_future_list[start+j][closest_obs_ind,:,0].numpy()
            #             closest_obs_y = test_future_list[start+j][closest_obs_ind,:,1].numpy()
                        
                        
            #             tvp_template['_tvp',ob, 'obsx'] = xpred[0,ob] + closest_obs_x[ob]
            #             tvp_template['_tvp',ob, 'obsy'] = ypred[0,ob] + closest_obs_y[ob]
            #             tvp_template['_tvp',ob, 'obsxn'] = xpred[0,ob+1] + closest_obs_x[ob+1]
            #             tvp_template['_tvp',ob, 'obsyn'] = ypred[0,ob+1] + closest_obs_y[ob+1]

            #             tensor_robot_loc_g = torch.empty((1, 1, 2)) 
            #             xcurr_stack = np.ones(1) * xcurr[-1] 
            #             ycurr_stack = np.ones(1) * ycurr[-1]
                        
            #             tensor_robot_loc_g[0,:,:] = torch.FloatTensor(np.column_stack((xcurr_stack.T,ycurr_stack.T)))
            #             test_goal_list_g[start][0,0,0] = xg
            #             test_goal_list_g[start][0,0,1] = yg
            #             test_goal_list[start+j] =  test_goal_list_g[start] - tensor_robot_loc_g
                        
            #             waypoint_x, waypoint_y , A_poly, b_poly, generator_vectors1 = pred_zono(3, model_pt, device, checkpoint, hyper_params, start+j, test_past_list, test_goal_list, test_waypoint_list, test_future_list, xcurr[-1].item(), ycurr[-1].item())

            
            # for n in range(5):
                # tensor_robot_loc = torch.empty((1, 8, 2))    

                # xcurr_stack = np.ones(8) * xpred[0,n] #xpred[0,1] #xcurr[-1]
                # ycurr_stack = np.ones(8) * ypred[0,n] #ypred[0,1] #ycurr[-1]
        
                
                # tensor_robot_loc[0,:,:] = torch.FloatTensor(np.column_stack((xcurr_stack.T,ycurr_stack.T)))

                # tvp_template['_tvp',n, 'ped_G'] = test_past_list_g[start+j+n][0].view(1,16).numpy().T
                # tvp_template['_tvp',:, 'goal_G'] = numpy.array([xg, yg])
                
                # test_past_list[start+j] =  test_past_list_g[start+j] - tensor_robot_loc
                # test_future_list[start+j] = test_future_list_g[start+j] - tensor_robot_loc

                # filtered = remove_tensors_over_distance(test_past_list[start+j], 4)
                # filtered_fut = remove_tensors_over_distance(test_future_list[start+j], 4)
            
                # test_past_list[start+j] = filtered
                # test_future_list[start+j] = filtered_fut

                # A_poly_no_ped, b_poly_no_ped, G, ATG_no_ped = no_ped(xpred, ypred)

                
                # c = np.array([xpred[0,n],  ypred[0,n]]).T
                # m = G.shape[1]  # Number of rows
                # binary_combinations = list(itertools.product([1, -1], repeat=m))
                # combinations_matrix = np.array(binary_combinations)
                # vertices0 = np.dot(combinations_matrix,G.T) + c
                # vertices0 = vertices0.reshape(vert_size)
                # vertices[:,n]=vertices0

                    
                # tvp_template['_tvp',:, 'vertices'] = vertices

                # if filtered.size(0) >= 1:                    
                #     obsx = filtered_fut[:,0,0].numpy()
                #     obsy = filtered_fut[:,0,1].numpy()
                #     dist = (obsx**2 + obsy**2)**0.5
                    
                #     closest_obs_ind = np.argmin(dist)
                #     closest_obs_x = test_future_list[start+j][closest_obs_ind,:,0].numpy()
                #     closest_obs_y = test_future_list[start+j][closest_obs_ind,:,1].numpy()
                    
                #     if n == 0:
                #         for h in range(3):
                #             tvp_template['_tvp',h, 'obsx'] = xcurr[-1] + closest_obs_x[h]
                #             tvp_template['_tvp',h, 'obsy'] = ycurr[-1] + closest_obs_y[h]
                #             tvp_template['_tvp',h, 'obsxn'] = xcurr[-1] + closest_obs_x[h+1]
                #             tvp_template['_tvp',h, 'obsyn'] = ycurr[-1] + closest_obs_y[h+1]
                    
                    

                    # tvp_template['_tvp',n, 'obs_loc'] = np.array([xpred[0,n], ypred[0,n]])

                    # tensor_robot_loc_g = torch.empty((1, 1, 2)) 
                    # xcurr_stack = np.ones(1) * xpred[0,n] #xcurr[-1] 
                    # ycurr_stack = np.ones(1) * ypred[0,n] #ycurr[-1]
                    
                    # tensor_robot_loc_g[0,:,:] = torch.FloatTensor(np.column_stack((xcurr_stack.T,ycurr_stack.T)))
                    # test_goal_list_g[start][0,0,0] = xg
                    # test_goal_list_g[start][0,0,1] = yg
                    # test_goal_list[start+j] =  test_goal_list_g[start] - tensor_robot_loc_g
                    
                    
                    # waypoint_x, waypoint_y , A_poly, b_poly, generator_vectorsnp, generator_matrices, generator_vectors, centers, ATG= pred_zono(3, model_pt, device, checkpoint, hyper_params, start+j, test_past_list, test_goal_list, test_waypoint_list, test_future_list, xpred[0,n].item(), ypred[0,n].item())
                    

                    # A_detach = A_poly.detach()
                    # b_detach = b_poly.detach()
                    # A_detach = np.array(A_detach.cpu().numpy())
                    # b_detach = np.array(b_detach.cpu().numpy())

                    ## Minkowski sum 
                    # ped_gen = torch.tensor([[0.3, 0.0],[0.0, 0.3]]).to(device)
                    # minkowski_G = torch.cat((generator_vectors,ped_gen.unsqueeze(0).unsqueeze(1).expand(1, 7, 2, 2)),dim = 2)
                    # A_m, b_m, L_m = H_rep(centers, minkowski_G, minkowski_G, xpred[0,n].item(), ypred[0,n].item(), device)
                    # print(minkowski_G.shape)
                    # minkowski_G_np = np.array(generator_matrices.detach().cpu().numpy())
              
                    # for z in range(7):
                    #     c = np.array([xpred[0,n]+waypoint_x[0],  ypred[0,n] + waypoint_y[0]]).T
                    #     m = minkowski_G_np[0,z].shape[0]  # Number of rows
                    #     binary_combinations = list(itertools.product([1, -1], repeat=m))
                    #     combinations_matrix = np.array(binary_combinations)
                    #     vertices0 = np.dot(combinations_matrix,minkowski_G_np[0,0]) + c
                    #     vertices0 = vertices0.reshape(vert_size)
                    #     vertices[:,n]=vertices0

                    
                    
                    # tvp_template['_tvp',:, 'vertices'] = vertices
                    
                  
                    # dx = waypoint_x[1:]-waypoint_x[:-1]
                    # dy = waypoint_y[1:]-waypoint_y[:-1]

                    
                    # speed = np.sqrt(dx**2 + dy**2)

                    # angles = np.arctan2(dy, dx)
                    # heading_change = angles[1:]-angles[:-1]
                    # vx = speed[1:] * np.cos(heading_change)
                
                    # x_n = torch.tensor([1.8+0.2*int_ind,4.50]).to(device)

                    # tvp_template['_tvp',n, 'obs_loc'] = np.array([6.8, 4.5])
                    # tvp_template['_tvp',n, 'obs_loc'] = np.array([1.8+0.2*(int_ind-1),4.50])

                    
                    

                    # pkk = 0
                    # for pk in range(test_future_list[start+j].shape[0]):
                    #     for zone_hor in range(6):
                            
                    #         # x_n = torch.tensor([xpred[0,n]+closest_obs_x[1+zone_hor], ypred[0,n]+closest_obs_y[1+zone_hor]]).to(device)
                    #         # x_n = torch.tensor([6.8, 4.5]).to(device)
                    #         x_n = torch.tensor([1.8+0.2*int_ind,4.50]).to(device)
                    #         # ped_gen = torch.tensor([[0.3,0.0],[0.0, 0.3]]).to(device)
                    #         # minkowski_G = torch.cat((generator_vectors,ped_gen.unsqueeze(0).unsqueeze(1).expand(1, 7, 2, 2)),dim = 2)
                    #         # A_m, b_m, L_m = H_rep(centers, minkowski_G, minkowski_G, xpred[0,n].item(), ypred[0,n].item(), device)
                    #         # x_n = torch.tensor([xpred[0,n]+test_future_list[start+j][pk,1+zone_hor,0].item(), ypred[0,n]+test_future_list[start+j][pk,1+zone_hor,1].item()]).to(device)
                            
                    #         # max = torch.max(torch.matmul(A_poly[zone_hor,:,:], x_n) - b_poly[zone_hor,:,:]).requires_grad_(True)
                    #         max = torch.max(torch.matmul(A_poly[zone_hor,:,:], x_n.unsqueeze(-1)) - b_poly[zone_hor,:,:]).requires_grad_(True)
                            
                    #         # max_i[pkk,0] = -max.item()
                    #         pkk=pkk+1
                    
                    ## Minkowski sum 
                    # ped_gen = torch.tensor([[0.3,0.0],[0.0, 0.3]]).to(device)
                    # minkowski_G = torch.cat((generator_vectors,ped_gen.unsqueeze(0).unsqueeze(1).expand(1, 7, 2, 2)),dim = 2)
                    # A_m, b_m, L_m = H_rep(centers, minkowski_G, minkowski_G, xpred[0,n].item(), ypred[0,n].item(), device)
                    # ped = torch.tensor([6.8, 4.5]).to(device)
                    
                    # for pk in range(test_future_list[start+j].shape[0]):
                    #     for zone_hor in range(5):
                    #         ## Minkowski Sum 
                    #         # A_m, b_m, G_m = mink_sum(xpred[0,n]+closest_obs_x[1+zone_hor].item(), 
                    #                                 #  ypred[0,n]+closest_obs_y[1+zone_hor].item(), 
                    #                                 #  generator_vectorsno[0,0])

                    #         # print(generator_vectors.shape)
                    #         # ped_gen = torch.tensor([[0.3,0.0],[0.0, 0.3]]).to(device)
                    #         # minkowski_G = torch.cat((generator_vectors,ped_gen.unsqueeze(0).unsqueeze(1).expand(1, 7, 2, 2)),dim = 2)
                    #         # A_m, b_m, L_m = H_rep(centers, minkowski_G, minkowski_G, xpred[0,n].item(), ypred[0,n].item(), device)
                    #         x_n = torch.tensor([1.8+0.2*int_ind,4.50]).to(device)
                    #         # x_n = torch.tensor([xpred[0,n]+test_future_list[start+j][pk,1+zone_hor,0].item(), ypred[0,n]+test_future_list[start+j][pk,1+zone_hor,1].item()]).to(device)
                    #         # A_m, b_m, G_m = mink_sum(xpred[0,n],ypred[0,n], 
                    #         #                          generator_vectorsnp[0,0])
                    #         # pred = np.array([6.5,4.5])
                    #         # print(A_m.shape)
                            
                    #         # max = np.max(np.dot(A_m,pred)-b_m)
                    #         # max = torch.max(torch.matmul(A_m[zone_hor,:,:], x_n) - b_m[zone_hor,:,:]).requires_grad_(True)
                    #         max = torch.max(torch.matmul(A_m[zone_hor,:,:], x_n.unsqueeze(-1)) - b_m[zone_hor,:,:]).requires_grad_(True)
                    #         # max = torch.max(torch.matmul(A_poly[zone_hor,:,:], x_n.unsqueeze(-1)) - b_poly[zone_hor,:,:]).requires_grad_(True)

                    #         max_i[pkk,0] = -max.item()
                    #         pkk=pkk+1
                    
                    # tvp_template['_tvp',n,'max_obs'] = max_i

                    # if ((test_goal_list[start+j][0,0,0].numpy())**2 + (test_goal_list[start+j][0,0,1].numpy())**2)**0.5 > 2.5:  
                    #     tvp_template['_tvp',n, 'xg_running'] = xpred[0,n] + waypoint_x[1] 
                    #     tvp_template['_tvp',n, 'yg_running'] = ypred[0,n] + waypoint_y[1] 

                        
                        
                    #     tvp_template['_tvp',n, 'vx_running'] = vx[0]

                    
                    #     waypoint_x_0 = waypoint_x[0+1]-waypoint_x[0] 
                    #     waypoint_y_0 = waypoint_x[0+1]-waypoint_y[0] 
                        
                    #     tvp_template['_tvp',n,'theta_g'] = angles[0+1] % (2*np.pi)

                    #     tvp_template['_tvp',n, 'A'] = A_detach[0,:,:] 
                    #     tvp_template['_tvp',n, 'b'] = b_detach[0,:,:] 

                        
                        
                    #     if n == 0:
                    #         tvp_template['_tvp',:, 'xg'] =  xpred[0,0] + waypoint_x[-1] 
                    #         tvp_template['_tvp',:, 'yg'] =   ypred[0,0] + waypoint_y[-1] 
                        
                    # else:
                    #     tvp_template['_tvp',:, 'stance'] = (-1)**(ind)
                    #     tvp_template['_tvp',:, 'xg_running'] = xg #test_goal_list_g[start][0,0,0].numpy() #test_goal_list_g[start][0,0,0].numpy() 
                    #     tvp_template['_tvp',:, 'yg_running'] = yg #test_goal_list_g[start][0,0,1].numpy() #test_goal_list_g[start][0,0,1].numpy()
                    #     waypoint_x_0 = -xcurr[-1]+ xg #test_goal_list_g[start][0,0,0].numpy() #test_waypoint_list_g[j][0][-1,0].numpy()
                    #     waypoint_y_0 = -ycurr[-1]+ yg #test_goal_list_g[start][0,0,1].numpy() #test_waypoint_list_g[j][0][-1,1].numpy()
                    #     orientation=  np.arctan2((waypoint_y_0),(waypoint_x_0))
                    #     tvp_template['_tvp',:,'theta_g'] = orientation % (2*np.pi)
                    #     tvp_template['_tvp',:, 'vx_running'] = 0
                    #     tvp_template['_tvp',:, 'vx_t'] = 0
                    #     tvp_template['_tvp',:,'theta_t'] = orientation % (2*np.pi)

                    #     tvp_template['_tvp',:, 'xg'] = xg # test_goal_list_g[start][0,0,0].numpy() #xcurr[-1]+waypoint_x[-1]#test_waypoint_list_g[j][0][-1,0].numpy()
                    #     tvp_template['_tvp',:, 'yg'] = yg # test_goal_list_g[start][0,0,1].numpy() #ycurr[-1]+waypoint_y[-1]#test_waypoint_list_g[j][0][-1,1].numpy()
                    #     tvp_template['_tvp',n, 'A'] =  0*A_poly_no_ped[n,0:14]
                    #     tvp_template['_tvp',n, 'b'] =  0*b_poly_no_ped[n,0:14]
                        
                        
                    
                    # tvp_template['_tvp',:, 'vx_t'] = 0
                    # tvp_template['_tvp',:,'theta_t'] = orientation % (2*np.pi)
                    
                # else:
                #     tvp_template['_tvp',:, 'xg'] = xg #test_goal_list_g[start][0,0,0].numpy() 
                #     tvp_template['_tvp',:, 'yg'] = yg #test_goal_list_g[start][0,0,1].numpy() 
                #     tvp_template['_tvp',:, 'xg_running'] = xg #test_goal_list_g[start][0,0,0].numpy() 
                #     tvp_template['_tvp',:, 'yg_running'] = yg #test_goal_list_g[start][0,0,1].numpy()
                #     tvp_template['_tvp',:, 'vx_running'] = 0
                #     tvp_template['_tvp',n, 'A'] = 0*A_poly_no_ped[n,0:14]
                #     tvp_template['_tvp',n, 'b'] =  0*b_poly_no_ped[n,0:14]

                    
                    
                
                
                        

                
        # else:
            # waypoint_x, waypoint_y , A_poly, b_poly, generator_vectorsnp, generator_matrices, generator_vectors, centers, ATG = pred_zono(3, model_pt, device, checkpoint, hyper_params, start+j, test_past_list, test_goal_list, test_waypoint_list, test_future_list, 0, 3)
            
            # A_detach = A_poly.detach()
            # b_detach = b_poly.detach()
            # A_detach = np.array(A_detach.cpu().numpy())
            # b_detach = np.array(b_detach.cpu().numpy())

            # tvp_template['_tvp',:, 'xg'] = xg# test_goal_list_g[start][0,0,0].numpy() # 0.2863 + waypoint_x[-1]#test_waypoint_list_g[1][0][-1,0].numpy()
            # tvp_template['_tvp',:, 'yg'] = yg#test_goal_list_g[start][0,0,1].numpy() # + waypoint_y[-1]#test_waypoint_list_g[1][0][-1,1].numpy()
            # tvp_template['_tvp',:, 'xg_running'] = xg#test_goal_list_g[start][0,0,0].numpy() #0.2863 + waypoint_x[-1]#test_waypoint_list_g[j][0][-1,0].numpy()
            # tvp_template['_tvp',:, 'yg_running'] = yg#test_goal_list_g[start][0,0,1].numpy() #3 + waypoint_y[-1]#test_waypoint_list_g[j][0][-1,1].numpy()
            # waypoint_x_0 = 0.2863 + waypoint_x[-1]#test_waypoint_list_g[1][0][-1,0].numpy()
            # waypoint_y_0 = 3 + waypoint_y[-1]#test_waypoint_list_g[1][0][-1,1].numpy()
            # tvp_template['_tvp',:, 'vx_running'] = 0
            # tvp_template['_tvp',:, 'A'] = 0*A_detach[0,:,:]  #A_poly_no_ped  # 0*A_poly[0,:,:]
            # tvp_template['_tvp',:, 'b'] = 0*b_detach[0,:,:]  #b_poly_no_ped  #0*np.ones(b_poly[0,:,:].shape)

            

        # tvp_template['_tvp',:, 'A'] = np.ones((14,2))
        # tvp_template['_tvp',:, 'b'] = np.ones((14,1))
        # tvp_template['_tvp',:, 'xg'] = test_goal_list_g[start][0,0,0].numpy() 
        # tvp_template['_tvp',:, 'yg'] = test_goal_list_g[start][0,0,1].numpy() 
        # tvp_template['_tvp',:, 'xg_running'] = test_goal_list_g[start][0,0,0].numpy() 
        # tvp_template['_tvp',:, 'yg_running'] = test_goal_list_g[start][0,0,1].numpy()
        # tvp_template['_tvp',:, 'vx_running'] = 0
        # tvp_template['_tvp',:, 'xg'] = xg 
        # tvp_template['_tvp',:, 'yg'] = yg 
        # tvp_template['_tvp',:, 'xg_running'] = xg 
        # tvp_template['_tvp',:, 'yg_running'] = yg
        # tvp_template['_tvp',:, 'vx_running'] = 0

        
        

    


            
            
            


            
            



                
            

            

        

        
        

        
        return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    
    mpc.set_nl_cons('zono', model.aux['zonotope_const'], 0.0)
    
    for i in range(ped_number):
        mpc.set_nl_cons(f'obs_zono_conts{i}', model.aux[f'obs_zono_conts{i}'], 0.0, soft_constraint=True, penalty_term_cons=1e2)

    # mpc.set_nl_cons('obs_zono_conts', model.aux['obs_zono_conts'], 0.0, soft_constraint=True, penalty_term_cons=1e2)
    # mpc.set_nl_cons('obs_zono_conts2', model.aux['obs_zono_conts2'], 0.0, soft_constraint=True, penalty_term_cons=1e2)
    # mpc.set_nl_cons('obs_zono_conts3', model.aux['obs_zono_conts3'], 0.0, soft_constraint=True, penalty_term_cons=1e2)
    # mpc.set_nl_cons('obs_zono_conts4', model.aux['obs_zono_conts4'], 0.0, soft_constraint=True, penalty_term_cons=1e2)
    # mpc.set_nl_cons('obs_zono_conts5', model.aux['obs_zono_conts5'], 0.0, soft_constraint=True, penalty_term_cons=1e2)
    # mpc.set_nl_cons('obs_zono_conts6', model.aux['obs_zono_conts6'], 0.0, soft_constraint=True, penalty_term_cons=1e2)

    
    



    mpc.setup()

    return mpc



   