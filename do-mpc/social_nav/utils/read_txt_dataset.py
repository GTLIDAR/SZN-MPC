import numpy as np
import numpy as np
# import visualization as vis
import pandas as pd
import torch
# import random
# import os
import pandas as pd
import matplotlib.pyplot as plt


def append_scene_from_txt_to_env(txt_filename):
    """Obtains and appends a scene from a txt file to the environment,
    also assigns the robot node id to a particular node in the environment
    :param txt_filename: string denoting the path of the txt file
    :param env: initialised environment to append the scene from the txt file to
    :param robot_node_id: robot node id that we are replacing the node in the environment to
    """

    scenes = []
    print('At', txt_filename)

    ### Read and parse the txt file
    data = pd.read_csv(txt_filename, sep='\t', index_col=False, header=None)
    data.columns = ['frame_id', 'track_id', 'x', 'y']
    data['frame_id'] = pd.to_numeric(data['frame_id'], downcast='integer')
    data['track_id'] = pd.to_numeric(data['track_id'], downcast='integer')

    data['frame_id'] = data['frame_id'] // 10

    data['frame_id'] -= data['frame_id'].min()

    # data['type'] = env.NodeType.PEDESTRIAN
    data['node_id'] = data['track_id'].astype(str) #assigning each node id to a particular track id
    data.sort_values('frame_id', inplace=True)

    # Mean Position
    data['x'] = data['x'] 
    data['y'] = data['y'] 

    max_timesteps = data['frame_id'].max()

    # # Initialising the scene object (most relevant object that we use)

    # scene = Scene(timesteps=max_timesteps+1, dt=0.4, name=txt_filename, aug_func=None)
    
    # # Obtain node data -> the position, velocity and acceleration of each node for their different timesteps
    # for node_id in pd.unique(data['node_id']):
    
    #     node_df = data[data['node_id'] == node_id]
    #     assert np.all(np.diff(node_df['frame_id']) == 1)
    #     #print(node_df)
    #     node_values = node_df[['x', 'y']].values

    #     if node_values.shape[0] < 2:
    #         continue

    #     new_first_idx = node_df['frame_id'].iloc[0]

    #     x = node_values[:, 0]
    #     y = node_values[:, 1]
    #     # vx = derivative_of(x, scene.dt)
    #     # vy = derivative_of(y, scene.dt)
    #     # ax = derivative_of(vx, scene.dt)
    #     # ay = derivative_of(vy, scene.dt)

    #     data_dict = {('position', 'x'): x,
    #                     ('position', 'y'): y}
    #     data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
    #     node_data = pd.DataFrame(data_dict, columns=data_columns)

    #     if node_id == robot_node_id:  #replace a human node in the scene with a robot node
    #         node = Node(node_type=env.NodeType.ROBOT, node_id=node_id, data=node_data)
    #         node.is_robot = True 
    #         scene.robot = node
        
    #     else:
    #         node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=node_id, data=node_data)
    #     node.first_timestep = new_first_idx

    #     scene.nodes.append(node)
    # scenes.append(scene)
    # env.scenes = scenes
    return data






def get_seq_data_mpc(textfile, batch_size, req_data_hist, req_future, radius):
    req_data = req_data_hist + req_future #required frames based on required past and future frames    
    
    ped_traj_tensor_new = torch.zeros((1, req_data_hist, 2)) 

    goal_tensor_new = torch.zeros((1, 1, 2))
    robot_tensor_new = torch.zeros((1, req_future, 2)) 

    ped_fttraj_tensor_new = torch.zeros((1, req_future, 2)) 
    ped_fttraj_tensor_new_g = ped_fttraj_tensor_new

    ped_traj_tensor_g_new = torch.zeros((1, req_data_hist, 2)) 
    goal_tensor_g_new = torch.zeros((1, 1, 2))
    robot_tensor_g_new = torch.zeros((1, req_future, 2))

    robot_node_id= np.array([])

    # batch_sum = 0

    batch_new = 0

    past_list =[]
    goal_list =[]
    waypoint_list = []
    future_list = []
    future_list_g = []
    past_list_g =[]
    goal_list_g =[]
    waypoint_list_g = []
    number_of_pred_list = []

    for txtpath in textfile:

        
        textfile_path = "../datatext/zara1/test/" + txtpath
        # textfile_path = "../datatext/univ/test/" + txtpath
        print(textfile_path)
        data = append_scene_from_txt_to_env(textfile_path)


        
        ## get nodes that are avaibale in frames > required data 
        for i in data['track_id'].values:
            framesid_trackid =data.loc[data['track_id'] == i]['frame_id'].values #frames in which robot_node_id exist
            if (data.loc[data['frame_id'] == framesid_trackid[0]]['track_id'].values).size > 2:
                if framesid_trackid.size > (req_data):
                    robot_node_id = np.append(robot_node_id, i)


        robot_node_id = np.unique(robot_node_id) #remove repteaded ID numbers
        # print(len(robot_node_id))
        # print(robot_node_id)
        np.random.shuffle(robot_node_id) #shuffle the the robot ID (to  get random robots)
        # print(robot_node_id)
        batch = 0
        # print(robot_node_id.shape)
        ped_in_radius_list = []
        print('number of available robots', len(robot_node_id))
        for r in robot_node_id: # loop through the nodes selected as robots

            # print(ped_in_radius_list)
            # print(np.any(np.array(ped_in_radius_list) > 1))
            if (np.any(np.array(ped_in_radius_list) > 1)):
                batch_new = 0
                # print('here')
            else:
                batch_new = - 1
                # print('dont count')
            batch = batch + 1 + batch_new
            print('batch:', batch)
            ped_in_radius_list = []
            # print(ped_traj_tensor_new.size(0))
            if batch > batch_size: # stop based on how many data bacthes wanted
                break
            else:
                # print('batch:', batch)
                # print('robot', r)
                # batch = batch + 1
            
                # r = 191.0 # 191.0 293.0 206.0
                # r = 163
                # print('robot: ', r)
                scene_robot  =data.loc[data['track_id'] == r] #all the scenes in which the robot exist
                # print('robot node id: ', r)

                

                scene_frames = scene_robot['frame_id'].values #all the frames in which the robot exist
                inc = 0

                # this loop was used to follow a single robot for multiple frame (uncomment if needed)
                # print('number of frames:', scene_frames.size-req_future - req_data_hist)
                for f in np.arange(req_data_hist, scene_frames.size-req_future, 1): #np.arange(req_data_hist, req_data_hist+1, 1): 
                
                
                # f = np.random.randint(req_data_hist, scene_frames.size-req_future) #random frame in the path of the robot 
                    # print('frame: ', f)
                    focused_ped = []
                    
                    
                    frame_0 = scene_frames[f-req_data_hist] # first frame
                    frame_f = scene_frames[f+req_future] # final frame in batch
                    frame_goal = scene_frames[-1] # goal frame in scene
                    # print('robot node id: ', r)
                    # print('initial frame: ',frame_0, ' final frame in batch: ', frame_f, 'goal frame: ', frame_goal)

                    for i in np.arange(frame_0, frame_f+1, 1): #append pedestrians in the req data frames
                        frames = data.loc[data['frame_id'] == i]
                        pedestrians = frames['track_id'].values
                        # print(pedestrians)
                        focused_ped = np.append(focused_ped, pedestrians) #pedestrians in the frame with the robot

                    # print(focused_ped)

                    pedestrians_0 = data.loc[data['frame_id'] == frame_0]['track_id'].values #pedestrians in the first frame
                    # print(pedestrians_0)

                    


                    avail_ped = []
                    for j in pedestrians_0: #focused on test pedestrians (only the pedestrauins visibile for more the req_data)
                        # print('node', j, 'exists this many times', np.count_nonzero(focused_ped == j, axis = 0))
                        if (np.count_nonzero(focused_ped == j, axis = 0)) >= req_data:
                            avail_ped.append(j)


                    # print('number of frames: ', req_data)
                    # print('number of pedestrians: ', pedestrians_0.size)
                    # print('number of pedestrians for more than N frames: ', len(test_ped))
                    # print('tested_ped: ', test_ped)  


                    # create robot trajectory tensor
                    r_x = []
                    r_y = []
                    r_x_g = []
                    r_y_g = []
                    robot_data = data.loc[data['track_id'] == r]
                    #robot (x,y) position in the current frame
                    r_x_0 = robot_data.loc[robot_data['frame_id'] == (frame_0 + req_data_hist)]['x'].values 
                    r_y_0 = robot_data.loc[robot_data['frame_id'] == (frame_0 + req_data_hist)]['y'].values

                    for j in np.arange(frame_0 + req_data_hist, frame_f, 1): #robot position from current to frame_f (future waypoints)
                            r_x = np.append(r_x,robot_data.loc[robot_data['frame_id'] == j]['x'].values - r_x_0)
                            r_y = np.append(r_y,robot_data.loc[robot_data['frame_id'] == j]['y'].values - r_y_0)
                            r_x_g = np.append(r_x_g,robot_data.loc[robot_data['frame_id'] == j]['x'].values)
                            r_y_g = np.append(r_y_g,robot_data.loc[robot_data['frame_id'] == j]['y'].values)
                            # print(j)

                    
                    
                    goal_tensor = torch.empty((1, 1, 2)) 
                    robot_tensor = torch.empty((1, req_future, 2))

                    goal_tensor_g = torch.empty((1, 1, 2)) 
                    robot_tensor_g = torch.empty((1, req_future, 2)) 
                    # goal position of the robot (relative to the initial robot position)
                    g_x = robot_data.loc[robot_data['frame_id'] == frame_goal]['x'].values - r_x_0
                    g_y = robot_data.loc[robot_data['frame_id'] == frame_goal]['y'].values  - r_y_0

                    g_x_g = robot_data.loc[robot_data['frame_id'] == frame_goal]['x'].values 
                    g_y_g = robot_data.loc[robot_data['frame_id'] == frame_goal]['y'].values

                    
                
                    avail_ped = np.setdiff1d(avail_ped, r) #remove robot Id from pedestrians
                    
                    ped_in_radius = []
                    for i in avail_ped: #create pedestrians data
                        x = []
                        y = []
                        ped_data = data.loc[data['track_id'] == i]
                        x_0 = ped_data.loc[ped_data['frame_id'] == frame_0+req_data_hist]['x'].values - r_x_0
                        y_0 = ped_data.loc[ped_data['frame_id'] == frame_0+req_data_hist]['y'].values - r_y_0

                        if (x_0**2 + y_0**2)**0.5 < 100: #limit to 3 meters
                            # print((x_0**2 + y_0**2)**0.5)
                            ped_in_radius.append(i)
                    
                    inc = 0
                    x = []
                    y = []
                    
                    ped_in_radius_list.append(len(ped_in_radius))
                    # print('number of peds: ', len(ped_in_radius))
                    if len(ped_in_radius) < 1:
                        # print('robot', r)
                        # print('no pedestrians within radius in this frame', f)
                        # batch = batch - 1
                        
                        # break
                        continue
                        # test_ped = avail_ped
                    else:
                        number_of_pred = len(ped_in_radius)
                        test_ped = ped_in_radius
                        # batch = batch + 1
                        # break
                        
                    
                        ped_traj_tensor = torch.empty((len(test_ped), req_data_hist, 2)) 
                        # goal_tensor = torch.empty(len(test_ped), 1, 2) 
                        # robot_tensor = torch.empty((len(test_ped), req_future, 2)) 

                        ped_fttraj_tensor = torch.empty((len(test_ped), req_future, 2)) 
                        ped_fttraj_tensor_g = ped_fttraj_tensor
                        ped_traj_tensor_g = torch.empty((len(test_ped), req_data_hist, 2)) 
                        # goal_tensor_g = torch.empty(len(test_ped), 1, 2) 
                        # robot_tensor_g = torch.empty((len(test_ped), req_future, 2)) 
                        
                        # create past trajectory tensors
                        for i in test_ped: #create pedestrians data
                            x = []
                            y = []
                            x_f = []
                            y_f = []
                            x_g = []
                            y_g = []
                            x_f_g = []
                            y_f_g = []
                            ped_data = data.loc[data['track_id'] == i]
                            for j in np.arange(frame_0 + req_data_hist , frame_0, -1): #pedestrians position from current frame to fraje_0
                                x = np.append(x,ped_data.loc[ped_data['frame_id'] == j]['x'].values - r_x_0)
                                y = np.append(y,ped_data.loc[ped_data['frame_id'] == j]['y'].values - r_y_0)
                                x_g = np.append(x_g,ped_data.loc[ped_data['frame_id'] == j]['x'].values )
                                y_g = np.append(y_g,ped_data.loc[ped_data['frame_id'] == j]['y'].values )
                            
                            for k in np.arange(frame_0 + req_data_hist, frame_f, 1): #pedestrians position from current frame to fraje_0
                                x_f = np.append(x_f,ped_data.loc[ped_data['frame_id'] == k]['x'].values - r_x_0)
                                y_f = np.append(y_f,ped_data.loc[ped_data['frame_id'] == k]['y'].values - r_y_0)
                                x_f_g = np.append(x_f_g,ped_data.loc[ped_data['frame_id'] == k]['x'].values)
                                y_f_g = np.append(y_f_g,ped_data.loc[ped_data['frame_id'] == k]['y'].values )
                            # print(inc)
                            ped_traj_tensor[inc,:,:] = torch.FloatTensor((np.column_stack((x,y)))) #pedestrians data (ped X history X xy)

                            ped_fttraj_tensor[inc,:,:] = torch.FloatTensor((np.column_stack((x_f,y_f)))) #pedestrians data (ped X future X xy)

                            ped_fttraj_tensor_g[inc,:,:] = torch.FloatTensor((np.column_stack((x_f_g,y_f_g)))) #pedestrians data (ped X future X xy)

                            # robot_tensor[inc,:,:] = torch.FloatTensor((np.column_stack((r_x,r_y)))) #robot data (ped X future X xy)
                            # goal_tensor[inc,:,:] = torch.FloatTensor(np.column_stack((g_x,g_y))) #goal data (ped X goal X xy)
                            
                            ped_traj_tensor_g[inc,:,:] = torch.FloatTensor((np.column_stack((x_g,y_g)))) #pedestrians data (ped X history X xy)
                            # robot_tensor_g[inc,:,:] = torch.FloatTensor((np.column_stack((r_x_g,r_y_g)))) #robot data (ped X future X xy)
                            # goal_tensor_g[inc,:,:] = torch.FloatTensor(np.column_stack((g_x_g,g_y_g))) #goal data (ped X goal X xy)
                            inc = inc +1
                        
                        robot_tensor[0,:,:] = torch.FloatTensor((np.column_stack((r_x,r_y)))) #robot data (ped X future X xy)
                        goal_tensor[0,:,:] = torch.FloatTensor(np.column_stack((g_x,g_y))) #goal data (ped X goal X xy)
                    
                    
                        robot_tensor_g[0,:,:] = torch.FloatTensor((np.column_stack((r_x_g,r_y_g)))) #robot data (ped X future X xy)
                        goal_tensor_g[0,:,:] = torch.FloatTensor(np.column_stack((g_x_g,g_y_g))) #goal data (ped X goal X xy)

                        #create a list
                        # torch.hstack((ped_traj_tensor_new, ped_traj_tensor))
                        # if append:
                        ped_traj_tensor_new = torch.cat((ped_traj_tensor_new,ped_traj_tensor),0) 
                        goal_tensor_new = torch.cat((goal_tensor_new,goal_tensor),0) 
                        robot_tensor_new = torch.cat((robot_tensor_new,robot_tensor),0) 
                        ped_fttraj_tensor_new = torch.cat((ped_fttraj_tensor_new,ped_fttraj_tensor),0) 
                        ped_fttraj_tensor_new_g = torch.cat((ped_fttraj_tensor_new_g,ped_fttraj_tensor_g),0) 
                        ped_traj_tensor_g_new = torch.cat((ped_traj_tensor_g_new,ped_traj_tensor_g),0) 
                        goal_tensor_g_new = torch.cat((goal_tensor_g_new,goal_tensor_g),0) 
                        robot_tensor_g_new = torch.cat((robot_tensor_g_new,robot_tensor_g),0)


                

                        ped_fttraj_tensor_new = ped_fttraj_tensor_new[1:,:,:]
                        ped_fttraj_tensor_new_g = ped_fttraj_tensor_new_g[1:,:,:]
                        goal_tensor_new = goal_tensor_new[1:,:,:]
                        robot_tensor_new = robot_tensor_new[1:,:,:]
                        ped_traj_tensor_g_new = ped_traj_tensor_g_new[1:,:,:]
                        goal_tensor_g_new = goal_tensor_g_new[1:,:,:]
                        robot_tensor_g_new = robot_tensor_g_new[1:,:,:]
                        ped_traj_tensor_new = ped_traj_tensor_new[1:,:,:]

                    if(ped_traj_tensor_g_new.size(0) > 1):
                        # print("ped:", ped_traj_tensor_g_new.size(0))
                        future_list.append(ped_fttraj_tensor_new)
                        future_list_g.append(ped_fttraj_tensor_new_g)
                        # past_list.append(ped_traj_tensor) 
                        goal_list.append(goal_tensor_new)
                        waypoint_list.append(robot_tensor_new)
                        past_list_g.append(ped_traj_tensor_g_new) 
                        goal_list_g.append(goal_tensor_g_new)
                        waypoint_list_g.append(robot_tensor_g_new)
                        # print(ped_traj_tensor_new.size())
                        past_list.append(ped_traj_tensor_new)
                        number_of_pred_list.append(number_of_pred)
                        batch_new = 0
                    # else:
                    #     print(ped_in_radius_list)
                    #     print(np.all(np.array(ped_in_radius_list)) <= 1)
                    #     if (np.all(np.array(ped_in_radius_list)) <= 1):
                    #         batch_new = - 1
                    #         print('here')
                        # continue
                        # print('here')

                    
                    
                    ped_traj_tensor_new = torch.zeros((1, req_data_hist, 2)) 
                    goal_tensor_new = torch.zeros((1, 1, 2))
                    robot_tensor_new = torch.zeros((1, req_future, 2)) 
                    ped_fttraj_tensor_new = torch.zeros((1, req_future, 2)) 
                    ped_fttraj_tensor_new_g = torch.zeros((1, req_future, 2)) 
                    ped_traj_tensor_g_new = torch.zeros((1, req_data_hist, 2)) 
                    goal_tensor_g_new = torch.zeros((1, 1, 2))
                    robot_tensor_g_new = torch.zeros((1, req_future, 2))

        



            # print(ped_traj_tensor.size())
            # print(robot_tensor.size())
            # print(goal_tensor.size())
        
    return past_list, goal_list, waypoint_list, future_list, past_list_g, goal_list_g, waypoint_list_g, future_list_g






