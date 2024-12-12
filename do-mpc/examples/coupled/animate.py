import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from casadi import *
from casadi.tools import *
import pdb
import sys
import time
import os
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull

sys.path.append('../../')
sys.path.append("../../social_nav/datatext/")
sys.path.append("../../social_nav/saved_models/")
sys.path.append("../../social_nav/scripts/")
sys.path.append("../../social_nav/utils/")
from pred_egonet import*
from pkl_gen import *
import do_mpc
import itertools

#from numpy.random import default_rng
# from template_model import template_model
# from template_mpc import template_mpc
# from template_simulator import template_simulator


from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse
from matplotlib.patches import Arrow
from matplotlib.patches import Rectangle
from do_mpc.data import save_results, load_results


def plot_polytope(center, generator_vectors, x, y, col,alpha, area, xlim=(0, 16), ylim=(0, 8)):
    
    c=center*0
    c[0] = center[0]+x
    c[1] = center[1]+y
    
    


    n = generator_vectors.shape[0]  # Number of rows
    m = 2  # Number of columns

    # Generate all possible combinations of +1, -1, and 0 in a binary matrix
    binary_combinations = list(itertools.product([1, -1], repeat=n))

    # Convert the binary combinations into a NumPy matrix
    combinations_matrix = np.array(binary_combinations)
    

    
    vertices = np.dot(combinations_matrix,generator_vectors) + c

    hull = ConvexHull(vertices)
    # if area:
    #     if hull.volume > 0.01:
    #         print(hull.volume)

    plt.fill(vertices[hull.vertices,0], vertices[hull.vertices,1], color=col, alpha=alpha, ec='k')
    # polygon = Polygon(vertices, True, edgecolor='k', facecolor='k' , alpha=0.2)

    
    return vertices

def generate_linear_trajectory_tensor(n, time_steps, start, finish):
    # Initialize the trajectory tensor with zeros
    trajectory_tensor = np.zeros((n, time_steps, 2))

    for example_idx in range(n):
        # Generate a linear interpolation from start to finish for x and y coordinates
        x_interp = np.linspace(start[0], finish[0], time_steps)
        y_interp = np.linspace(start[1], finish[1], time_steps)

        # Store the interpolated coordinates in the tensor
        trajectory_tensor[example_idx, :, 0] = x_interp + example_idx
        trajectory_tensor[example_idx, :, 1] = y_interp 

    return torch.from_numpy(trajectory_tensor)

#results = load_results('./results/heading constraint.pkl')
# test_past_list, test_goal_list, test_waypoint_list, test_past_list_g, test_goal_list_g, test_waypoint_list_g = get_seq_data('/home/ashamsah3/human_prediction/do-mpc/social_nav/datatext/students003_val.txt', 1, 8,6)

batch_size = 20
req_data_hist = 8
req_future = 8
radius = 4
train = 0
dataset = "Zara1_mpc"
# dataset = "students003_mpc"

test_past_list, test_goal_list, test_waypoint_list, test_future_list, test_past_list_g, test_goal_list_g, test_waypoint_list_g, test_future_list_g = read_from_pkl(dataset, batch_size, req_data_hist, req_future, radius, 0, dataset)
 


# results_digitstl = load_results('./results/086_zono.pkl')
# results_digitstl = load_results('./results/001_GP.pkl')
results_digitstl = load_results('./results/coupled_w_social_5.pkl')

ped_number = 5

x = results_digitstl['mpc']['_x','x_g']
y = results_digitstl['mpc']['_x','y_g']
heading = results_digitstl['mpc']['_x','d_theta']

x_g_digit = results_digitstl['mpc']['_x','x_g_digit']
y_g_digit = results_digitstl['mpc']['_x','y_g_digit']
xg = results_digitstl['mpc']['_tvp','xg']
yg = results_digitstl['mpc']['_tvp','yg']



xd_com = results_digitstl['mpc']['_x','xd_com']
# obs_loc =results_digitstl['mpc']['_tvp','obs_loc']

steps = x.shape[0]
horz = 4

NN_output= results_digitstl['mpc']['_aux','NN']

NN_output = NN_output.reshape(steps,10,7)
NN_output = NN_output.transpose(0,2,1)

center_out = results_digitstl['mpc']['_aux','center_out']

center = NN_output[:,0, 0:2] 
G = NN_output[:,0, 2:].reshape(steps,4,2)

# xg = results_digitstl['mpc']['_tvp','xg_running',0]
# yg = results_digitstl['mpc']['_tvp','yg_running',0]


# social_waypoint = results_digitstl['mpc']['_aux','social_waypoint']

zonotope_const = results_digitstl['mpc']['_aux','zonotope_const']
# print(zonotope_const)
fig = plt.figure(figsize=(14,9))
ax1 = fig.add_subplot(1,1,1)
circle = Circle((0, 0), 4, alpha=0.5)
border = Rectangle((-0.999999,-0.9), 13.9, 13.9, fc = "None", ec="black" )
ax1.set_xlim([-1, 13])
ax1.set_ylim([-1, 13])
plt.pause(0.1)

start = 1

obs_zono = Rectangle((-0.14,-0.15), 0.3, 0.3, fc = "None", ec="black" )

ped = 4  # Number of examples
time_steps = 100  # Number of time steps
st = np.array([3.0, 0.0])  # Start point (x, y)
finish = np.array([3.0, 16.0])  # Finish point (x, y)
# st = np.array([8.0, 8.0])  # Start point (x, y)
# finish = np.array([3.0, 0.0])
pedestrians = generate_linear_trajectory_tensor(ped, time_steps, st, finish)

for i in range(3,250,1):
#     t = time.time()
    ax1.plot(x,y, alpha = 0.5, color='black')
    # i = 57
    ax1.scatter(x[i],y[i], color ='blue', alpha = 0.5)
    # ax1.scatter(x_g_digit[i],y_g_digit[i], color ='red', alpha = 0.5)
    # ax1.scatter(social_waypoint[i,0],social_waypoint[i,1], color='r')
    # social_waypoint_pred = results_digitstl['mpc'].prediction(('_aux','social_waypoint'), t_ind=i)
    # ax1.scatter(social_waypoint_pred[0,:],social_waypoint_pred[1,:], color='m')
    # zonotope_const = results_digitstl['mpc'].prediction(('_aux','zonotope_const'), t_ind=i)
    # print(np.max(zonotope_const))

    # xd_com = results_digitstl['mpc'].prediction(('_x','xd_com'), t_ind=i)
    # print(xd_com)

    x_pred = results_digitstl['mpc'].prediction(('_x','x_g'), t_ind=i)
    y_pred = results_digitstl['mpc'].prediction(('_x','y_g'), t_ind=i)
    
    # ax1.scatter(x_pred,y_pred, color ='red', alpha = 0.5)


    NN_pred = results_digitstl['mpc'].prediction(('_aux','NN'), t_ind=i)
    NN_pred = NN_pred.reshape(70,horz)
    
    # NN_pred = NN_pred.reshape(10,7,horz,1)
    NN_pred = NN_pred.T
    center_pred = NN_pred[:,0:2] 
   

    
    # ax1.scatter(center_pred[0,:,0]+x_pred[0,1:,0],center_pred[1,:,0]+y_pred[0,1:,0], color='b')
    # ax1.scatter(center_out[i,0],center_out[i,1], color='k')
    G_pred = NN_pred[:,2:10].reshape(horz,4,2) 

    center_out_pred = results_digitstl['mpc'].prediction(('_aux','center_out'), t_ind=i)

    xg_running = results_digitstl['mpc'].prediction(('_tvp','xg_running'), t_ind=i+1)
    yg_running = results_digitstl['mpc'].prediction(('_tvp','yg_running'), t_ind=i+1)
    ax1.scatter(xg_running[0,:,0],yg_running[0,:,0], color='k')

    # plot_polytope(center_out[i],G[i],0,0,'r')
    
    # plot_polytope(center[i],G[i],x[i],y[i],'b')
    # print('center_out[i]', center_out[i,0])
    # print('center[i]',center[i,0]+x[i])
    # print('x_pred',x_pred[0,1,0])

    # for n  in range(horz):
    #     for j in range(6):
    #         center_range = np.arange(10*j+0,2+10*j)
    #         G_range = np.arange(10*j+2,10+10*j)
    #         center_pred_all = NN_pred[:,center_range] 
    #         G_pred_all = NN_pred[:,G_range].reshape(horz,4,2) 
    #         plot_polytope(center_pred_all[n,:],G_pred_all[n,:,:],x_pred[0,n],y_pred[0,n],'r')
    # for j in range(7):
    #     center_range = np.arange(10*j+0,2+10*j)
    #     G_range = np.arange(10*j+2,10+10*j)
    #     center_pred_all = NN_pred[:,center_range] 
    #     G_pred_all = NN_pred[:,G_range].reshape(horz,4,2) 
    #     plot_polytope(center_pred_all[0,:],G_pred_all[0,:,:],x_pred[0,0],y_pred[0,0],'r')

    # for j in range(7):
    #     center_range = np.arange(10*j+0,2+10*j)
      #     G_range = np.arange(10*j+2,10+10*j)
    #     center_pred_all = NN_pred[:,center_range] 
    #     G_pred_all = NN_pred[:,G_range].reshape(horz,4,2) 
    #     plot_polytope(center_pred_all[1,:],G_pred_all[1,:,:],x_pred[0,1],y_pred[0,1],'b')

    vector1 = np.array([0.15, 0])
    vector2 = np.array([0, 0.15])

    # Create the 3D array
    result_array = np.array([
        [vector1, vector2],
        [vector1, vector2],
        [vector1, vector2],
        [vector1, vector2]
    ])
    
    GP_G = results_digitstl['mpc'].prediction(('_tvp','GP_G')).reshape(1,horz+1,2,2)
    # print(GP_G.shape)
    GP_G= np.swapaxes(GP_G, 0, 1)
    GP_pred = np.concatenate((G_pred, GP_G[1:,0,:,:],result_array), axis=1)
    # print(GP_G[1:,:,:].shape)
    for n in range(horz):
        plot_polytope(center_out_pred[:,n,0],GP_pred[n,:,:],0,0,'r',0.2,0)

    for n in range(horz):
        # plot_polytope(center_pred[n,:],G_pred[n,:,:],x_pred[0,n+1],y_pred[0,n+1],'b')
        plot_polytope(center_out_pred[:,n,0],G_pred[n,:,:],0,0,'c',0.2,0)
        
    # print(np.max(np.sqrt(GP_pred[:,:,0]**2 + GP_pred[:,:,1]**2)))
    # sum_G=np.sum(GP_pred,axis=1)
    # print(np.max(np.sqrt(sum_G[:,0]**2 + sum_G[:,1]**2)))
        
    

    # # vert[i,:,0] = vert[i,:,0]+x[i]
    # # vert[i,:,1] = vert[i,:,1]+y[i]
    # hull = ConvexHull(vert[i])
    # ax1.fill(vert[i,hull.vertices,0], vert[i,hull.vertices,1], 'b', alpha=0.2)

    # for n in range(3):
    #     hull = ConvexHull(vert_pred[:,:,n,0])
    #     ax1.fill(vert_pred[hull.vertices,0,n,0], vert_pred[hull.vertices,1,n,0], 'b', alpha=0.2)


    # ax1.scatter(x_pred[0,1:,0],y_pred[0,1:,0],color='y')


    # for j in range(test_past_list_g[start+i].size(0)):
    #     ax1.scatter(test_past_list_g[start+i][j,0,0],test_past_list_g[start+i][j,0,1], color = 'green')
    #     ax1.plot(test_past_list_g[start+i][j,:,0],test_past_list_g[start+i][j,:,1], color = 'green', linestyle = '--')
    #     # ax1.plot(test_future_list_g[start+i][j,:,0],test_future_list_g[start+i][j,:,1], color = 'red', linestyle = '--')
    # for p in range(4):
    #     ax1.scatter(pedestrians[p,i+10,0],pedestrians[p,i+10,1], color = 'g')
        

    ax1.scatter(12, 6, marker="x", color = 'gold', s = 180)
    # ax1.scatter(xg[i], yg[i], marker="x", color = 'k', s = 180)


    ##---------------   Plot pedestrians from TVP
    for ped in range(ped_number):
        obs_loc_pred = results_digitstl['mpc'].prediction(('_tvp',f'obs_loc{ped}'), t_ind=i)
        obs_loc_pred = obs_loc_pred.reshape(1,horz+1,2)
        ped_G1 = results_digitstl['mpc'].prediction(('_tvp',f'ped_G{ped}1'), t_ind=i).reshape(1,horz+1,2)
        ped_G2 = results_digitstl['mpc'].prediction(('_tvp',f'ped_G{ped}2'), t_ind=i).reshape(1,horz+1,2)
        ped_G3 = results_digitstl['mpc'].prediction(('_tvp',f'ped_G{ped}3'), t_ind=i).reshape(1,horz+1,2)
        ped_G4 = results_digitstl['mpc'].prediction(('_tvp',f'ped_G{ped}4'), t_ind=i).reshape(1,horz+1,2)

        ped_G = np.concatenate((ped_G1, ped_G2, ped_G3, ped_G4), axis=0)
        # sum_G=np.sum(ped_G,axis=0)
        
        if (obs_loc_pred[0,0,0] == 0.0 and obs_loc_pred[0,0,1] == 0.0):
            k = 1 
        # else:
        #     for n in range(horz):
                # plot_polytope(obs_loc_pred[0,n,:],ped_G[:,n,:],0,0,'g',0.2,1)
        #     # ax1.plot(obs_loc_pred[0,:,0], obs_loc_pred[0,:,1], color='g')
        #     ax1.scatter(obs_loc_pred[0,0,0], obs_loc_pred[0,0,1], color='g')

    for ped in range(ped_number):
        obs_loc_pred = results_digitstl['mpc'].prediction(('_tvp',f'obs_loc{ped}g'), t_ind=i)
        obs_loc_pred = obs_loc_pred.reshape(1,horz+1,2)
        
        ax1.scatter(obs_loc_pred[0,0,0], obs_loc_pred[0,0,1], color='g', alpha=0.3)


    ##---------------   Plot pedestrians from NN coupled
    NN_ped_init = results_digitstl['mpc'].prediction(('_tvp','ped_init'), t_ind=i)
    NN_ped_past = results_digitstl['mpc'].prediction(('_tvp','ped_past'), t_ind=i)
    NN_ped_past = NN_ped_past.reshape(horz+1,8,2,ped_number)
    NN_ped_init=NN_ped_init.reshape(horz+1,2,ped_number)
    # NN_ped_init=NN_ped_init.reshape(1,horz+1,2,11,1)
    NN_ped = results_digitstl['mpc'].prediction(('_aux','NN_ped'), t_ind=i)
    # print(NN_ped)
    NN_ped =NN_ped.reshape(ped_number,10,7,horz)


    peds_centers = results_digitstl['mpc'].prediction(('_aux','peds_centers'), t_ind=i).reshape(16,ped_number+1,horz,1)
    # print(peds_centers[:,1,:,0])
    # print(peds_centers.shape)
    # NN_ped =NN_ped.reshape(11,7,10,horz,1)
    # ped_centers_NN = NN_ped[:,0,:2,0,0]
   
    
    for ped in range(ped_number): 
        ped_G1 = NN_ped[ped,2:4,:5,0].reshape(1,2,5)
        ped_G2 = NN_ped[ped,4:6,:5,0].reshape(1,2,5)
        ped_G3 = NN_ped[ped,6:8,:5,0].reshape(1,2,5)
        ped_G4 = NN_ped[ped,8:,:5,0].reshape(1,2,5)

        nn_obs_loc = results_digitstl['mpc'].prediction(('_aux',f'nn_obs_loc{ped}'), t_ind=i)

        G_z = results_digitstl['mpc'].prediction(('_aux',f'G{ped}'), t_ind=i).reshape(4,2,horz)
        
        # ped_G1 = results_digitstl['mpc'].prediction(('_tvp',f'ped_G{ped}1'), t_ind=i).reshape(1,horz+1,2)
        # ped_G2 = results_digitstl['mpc'].prediction(('_tvp',f'ped_G{ped}2'), t_ind=i).reshape(1,horz+1,2)
        # ped_G3 = results_digitstl['mpc'].prediction(('_tvp',f'ped_G{ped}3'), t_ind=i).reshape(1,horz+1,2)
        # ped_G4 = results_digitstl['mpc'].prediction(('_tvp',f'ped_G{ped}4'), t_ind=i).reshape(1,horz+1,2)

       
        for n in range(horz):
            plot_polytope(nn_obs_loc[:,n,0],G_z[:,:,n],0,0,'g',0.2,1)
            # ax1.scatter(peds_centers[2,ped,:,0],peds_centers[3,ped,:,0],color='r')

            
            # ax1.scatter(NN_ped_past[1,:,0,1]+x[i],NN_ped_past[1,:,1,1]+y[i],color='r')
            # ax1.scatter(NN_ped_past[n,-1,0,ped]+x[i],NN_ped_past[n,-1,1,ped]+y[i],color='k')
            # ax1.scatter(nn_obs_loc[0,n],nn_obs_loc[1,n],color='b')
    
    

    

    # print(ped_G[i])

    # for n in range(horz):
    #     ax1.scatter(obs_loc_pred[0,0,0], obs_loc_pred[0,0,1], color='g')
    #     plot_polytope(obs_loc_pred[0,n,:],ped_G1[i,:,:],0,0,'g')
    #     ax1.scatter(obs_loc_pred2[0,0,0], obs_loc_pred2[0,0,1], color='g')
    #     plot_polytope(obs_loc_pred2[0,n,:],ped_G2[i,:,:],0,0,'g')
    #     ax1.scatter(obs_loc_pred3[0,0,0], obs_loc_pred3[0,0,1], color='g')
    #     plot_polytope(obs_loc_pred3[0,n,:],ped_G3[i,:,:],0,0,'g')

    #     ax1.scatter(obs_loc_pred4[0,0,0], obs_loc_pred4[0,0,1], color='g')
    #     plot_polytope(obs_loc_pred4[0,n,:],ped_G4[i,:,:],0,0,'g')
    #     ax1.scatter(obs_loc_pred5[0,0,0], obs_loc_pred5[0,0,1], color='g')
    #     plot_polytope(obs_loc_pred5[0,n,:],ped_G5[i,:,:],0,0,'g')
    #     ax1.scatter(obs_loc_pred6[0,0,0], obs_loc_pred6[0,0,1], color='g')
    #     plot_polytope(obs_loc_pred6[0,n,:],ped_G6[i,:,:],0,0,'g')

    circle = Circle((x[i],y[i]), 4, alpha=1, fc = 'none', ec='k', ls='--')
    ax1.add_patch(circle)
    ax1.add_patch(border)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')

    arrow = Arrow(x[i],y[i],0.15*cos(heading[i]),0.15*sin(heading[i]), width=0.2, fc="blue")
    ax1.add_patch(arrow)


    plt.gca().set_aspect('equal')

    ax1.set_xlim([-1, 13])
    ax1.set_ylim([-1, 13])
    dpi = 700
    plt.rcParams['figure.dpi'] = dpi
    plt.draw()

    plt.axis('off')

    plt.pause(0.01)

    plt.cla()



