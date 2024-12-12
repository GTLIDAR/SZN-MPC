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
 



results_digitstl = load_results('./results/standing.pkl')

ped_number = 5

# 83 is cbf zono, 30 pedestrians

x = results_digitstl['mpc']['_x','x_g']
y = results_digitstl['mpc']['_x','y_g']
heading = results_digitstl['mpc']['_x','d_theta']

uheading = results_digitstl['mpc']['_u','u_d_theta']

pf_x = results_digitstl['mpc']['_u','pf_x']


d_heading = results_digitstl['mpc']['_tvp','theta_t']

running_heading = results_digitstl['mpc']['_tvp','theta_g']


x_g_digit = results_digitstl['mpc']['_x','x_g_digit']
y_g_digit = results_digitstl['mpc']['_x','y_g_digit']
xg = results_digitstl['mpc']['_tvp','xg']
yg = results_digitstl['mpc']['_tvp','yg']


xd_com = results_digitstl['mpc']['_x','xd_com']
# obs_loc =results_digitstl['mpc']['_tvp','obs_loc']

steps = x.shape[0]
horz = 6

NN_output= results_digitstl['mpc']['_aux','NN']

NN_output = NN_output.reshape(steps,10,7)
NN_output = NN_output.transpose(0,2,1)

center_out = results_digitstl['mpc']['_aux','center_out']

center = NN_output[:,0, 0:2] 
G = NN_output[:,0, 2:].reshape(steps,4,2)

x_social = results_digitstl['mpc']['_tvp','xg_running',0]
y_social = results_digitstl['mpc']['_tvp','yg_running',0]


# social_waypoint = results_digitstl['mpc']['_aux','social_waypoint']

zonotope_const = results_digitstl['mpc']['_aux','zonotope_const']
# print(zonotope_const)
fig = plt.figure(figsize=(14,9))
ax1 = fig.add_subplot(1,1,1)
circle = Circle((0, 0), 4, alpha=0.5)

border = Rectangle((-0.999999,-0.9), 13.9, 13.9, fc = "None", ec="black" )
ax1.set_xlim([-1, 13])
ax1.set_ylim([-1, 13])
plt.pause(3)

# ax1.plot(heading)
# ax1.plot(uheading)
# plt.show()

start = 1

obs_zono = Rectangle((-0.14,-0.15), 0.3, 0.3, fc = "None", ec="black" )

ped = 4  # Number of examples
time_steps = 100  # Number of time steps
st = np.array([3.0, 0.0])  # Start point (x, y)
finish = np.array([3.0, 16.0])  # Finish point (x, y)
# st = np.array([8.0, 8.0])  # Start point (x, y)
# finish = np.array([3.0, 0.0])
pedestrians = generate_linear_trajectory_tensor(ped, time_steps, st, finish)

def alternate_array(n):
    result = []
    for i in range(n):
        if i % 2 == 0:
            result.append(1)
        else:
            result.append(-1)
    return result

n = x.shape[0]  # Change this to the desired size of the array
stance = alternate_array(n)

def multiply_array(arr, constant):
    return [element * constant for element in arr]

foot_y_local = multiply_array(np.array(stance),0.25)
foot_yy = np.array(foot_y_local).reshape(-1,1)
foot_xx = pf_x

foot_x = foot_xx* np.cos(heading) + foot_yy*np.sin(heading)
foot_y = foot_xx* np.sin(heading) +  foot_yy*np.cos(heading)
for i in range(1,100,1):
    # 12 ,38, 67
    # i = 80
    #     # i = 45
    #     # i = 37
    # i = 34
    #     t = time.time()
    # i = 0
    # ax1.plot(x,y, alpha = 0.5, color='blue')
    # i = 57

    ax1.scatter(x[:i],y[:i], alpha = 0.5, color='blue')
    ax1.scatter(x[:i]+foot_xx[:i], y[:i]+foot_yy[:i], color = 'k', marker='x')
    # ax1.scatter(x_social,y_social, color ='black', alpha = 1)


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



    NN_pred = results_digitstl['mpc'].prediction(('_aux','NN'), t_ind=i)
    NN_pred = NN_pred.reshape(70,horz)

    # NN_pred = NN_pred.reshape(10,7,horz,1)
    NN_pred = NN_pred.T
    center_pred = NN_pred[:,0:2] 



    # ax1.scatter(center_pred[0,:,0]+x_pred[0,1:,0],center_pred[1,:,0]+y_pred[0,1:,0], color='b')
    # ax1.scatter(center_out[i,0],center_out[i,1], color='k')
    G_pred = NN_pred[:,2:10].reshape(horz,4,2) 

    center_out_pred = results_digitstl['mpc'].prediction(('_aux','center_out'), t_ind=i)

    xg_running = results_digitstl['mpc'].prediction(('_tvp','xg_running'), t_ind=i)
    yg_running = results_digitstl['mpc'].prediction(('_tvp','yg_running'), t_ind=i)
    # ax1.plot(xg_running[0,:,0],yg_running[0,:,0], color='r')



    vector1 = np.array([0.15, 0])
    vector2 = np.array([0, 0.15])

    # Create the 3D array
    result_array = np.array([
        [vector1, vector2],
        [vector1, vector2],
        [vector1, vector2],
        [vector1, vector2],

        [vector1, vector2],
        [vector1, vector2]
    ])

    GP_G = results_digitstl['mpc'].prediction(('_tvp','GP_G')).reshape(1,horz+1,2,2)
    # print(GP_G.shape)
    GP_G= np.swapaxes(GP_G, 0, 1)
    # print(GP_G.shape)
    GP_pred = np.concatenate((G_pred, GP_G[1:,0,:,:],result_array), axis=1)
    # print(GP_G[1:,:,:].shape)
    for n in range(horz):
        plot_polytope(center_out_pred[:,n,0],GP_pred[n,:,:],0,0,'c',0.5,0)

    # for n in range(horz):
        # plot_polytope(center_pred[n,:],G_pred[n,:,:],x_pred[0,n+1],y_pred[0,n+1],'b')
        # plot_polytope(center_out_pred[:,n,0],G_pred[n,:,:],0,0,'c',0.2,0)
    ax1.scatter(x_pred[0,:,0],y_pred[0,:,0], color ='b')

    #
    # ax1.scatter(12, 6, marker="x", color = 'gold', s = 180)

    ax1.scatter(10, 10, marker="x", color = 'gold', s = 180)

    ##---------------   Plot pedestrians from TVP
    # ped_number = 30
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
        else:
            for n in range(horz):
                plot_polytope(obs_loc_pred[0,n,:],ped_G[:,n,:],0,0,'g',0.5,1)
            # ax1.plot(obs_loc_pred[0,:,0], obs_loc_pred[0,:,1], color='g')
            ax1.scatter(obs_loc_pred[0,0,0], obs_loc_pred[0,0,1], color='g')

    for ped in range(ped_number):
        obs_loc_pred = results_digitstl['mpc'].prediction(('_tvp',f'obs_loc{ped}g'), t_ind=i)
        obs_loc_pred = obs_loc_pred.reshape(1,horz+1,2)

        obs_loc_predg = results_digitstl['mpc']['_tvp',f'obs_loc{ped}g']
        
        ax1.scatter(obs_loc_pred[0,0,0], obs_loc_pred[0,0,1], color='g', alpha=0.3)
        # ax1.scatter(obs_loc_predg[i,0], obs_loc_predg[i,1], color='k', alpha=1)

    ax1.scatter(x[i],y[i], color ='blue', alpha = 1)

    circle = Circle((x[i],y[i]), 4, alpha=1, fc = 'none', ec='k', ls='--')
    ax1.add_patch(circle)
    ax1.add_patch(border)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    arrow = Arrow(x[i],y[i],cos(heading[i]),sin(heading[i]), width=0.2, fc="blue", ec="black")
    ax1.add_patch(arrow)
    arrow = Arrow(x[i],y[i],0.5*cos(d_heading[i]),0.5*sin(d_heading[i]), width=0.2, fc="red")
    # ax1.add_patch(arrow)
    arrow = Arrow(x[i],y[i],cos(running_heading[i]),sin(running_heading[i]), width=0.2, fc="red", ec="black")
    # ax1.add_patch(arrow)

    ## ----------- Social metric -----------#
    social_metric = (x_pred[0,:,0] - xg_running[0,:,0])**2 +  (y_pred[0,:,0] - yg_running[0,:,0])**2 +  (heading[i] -running_heading[i])**2 
    # print(social_metric)

    # plt.title('frame %i' %i)

    plt.gca().set_aspect('equal')

    ax1.set_xlim([-1, 13])
    ax1.set_ylim([-1, 13])
    dpi = 700
    plt.rcParams['figure.dpi'] = dpidpi = 700
    plt.rcParams['figure.dpi'] = dpi
    plt.draw()


    # plt.savefig('001_digit_iros_7.svg', format='svg')
    # plt.show()
    plt.axis('off')
    # plt.title('frame %i' %i)

    # plt.show()

    plt.pause(0.1)

    plt.cla()


