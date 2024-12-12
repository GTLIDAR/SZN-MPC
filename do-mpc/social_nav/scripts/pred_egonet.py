import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader
import argparse
sys.path.append("../eth_ucy/")


sys.path.append("utils/")
import yaml
from models import *
import numpy as np
from read_txt_dataset import *
import time





def run(ped_traj_tensor, goal_tensor, model, device, hyper_params, best_of_n = 1, ):
    model.eval()
    assert best_of_n >= 1 and type(best_of_n) == int
    
    with torch.no_grad():
                
        x = ped_traj_tensor
        y = goal_tensor
        y = y.cpu().numpy()
        
        
        # reshape the data
        x = x.view(-1, x.shape[1]*x.shape[2])
        x = x.to(device)

        dest = torch.from_numpy(y[:, -1, :]).to(device)

        all_guesses = []
        
        for _ in range(best_of_n):

            generated_waypoint = model.forward(x, dest, device=device)
            generated_waypoint = generated_waypoint.cpu().numpy()
            all_guesses.append(generated_waypoint)
       
        
        all_guesses = np.array(all_guesses)
        all_pred = np.reshape(all_guesses,(-1, hyper_params['future_length'], 2))
      

    return all_pred


def run_sum(ped_traj_tensor, goal_tensor, future_tensor, hyper_params, model, device, best_of_n = 10):
    model.eval()
    assert best_of_n >= 1 and type(best_of_n) == int
    
    with torch.no_grad():
                
        x = ped_traj_tensor.double()
        y = goal_tensor.double().view(1,1,2).double()
        y = y.cpu().numpy()

        # reshape the data
        x = x.view(-1, x.shape[1]*x.shape[2])
        x = x.to(device)
        # print(x.size())

        ptraj = model.sum(x, x.size(0))
        # ptraj_batch[example_no,:] = ptraj

        dest = torch.from_numpy(y[:, -1, :]).to(device)

        all_guesses = []
        all_features = []

        all_generated_wayponts = torch.zeros(best_of_n, 2*hyper_params['future_length'])
        
        
        for _ in range(best_of_n):

            generated_waypoint = model.forward(ptraj, dest, device=device)
            all_generated_wayponts[_,:]=generated_waypoint
            generated_waypointnp = generated_waypoint.cpu().numpy()
            all_guesses.append(generated_waypointnp)
       
        
        all_guesses = np.array(all_guesses)
        all_pred = np.reshape(all_guesses,(-1, hyper_params['future_length'], 2))

        
        
        ################### STL robustness ######################
        
        # xy_generated_waypoint = torch.mean(all_generated_wayponts, dim = 0).view(-1, hyper_params['future_length'], 2)
        # # for i in np.arange(0, xy_generated_waypoint.size(0),1):
            
        # #     generated_waypoint_batch[past_size_list[i]:past_size_list[i+1],:,:] = xy_generated_waypoint[i,:,:]
           

        # generated_waypoint_batch = xy_generated_waypoint.to(device)
        
        # # print('generated waypoint size: ', generated_waypoint_batch)
        # # print('future_tensor: ', future_tensor)
        # # print(generated_waypoint_batch.size())
        # # print(future_tensor.size())
        # # xy_ped_0_diff = generated_waypoint_batch*0
        # # Expand the dimensions of tensors to be broadcastable
        # # expanded_generated = generated_waypoint_batch.unsqueeze(1)  # shape: (n, 1, 8, 2)
        # # expanded_future = future_tensor.to(device).unsqueeze(0)  # shape: (1, f, 8, 2)
        # # print(expanded_generated.size(), expanded_future.size())
        # # Compute the difference and reshape to the desired size
        # # xy_ped_0_diff = expanded_generated - expanded_future  # shape: (n, f, 8, 2)
        # xy_ped_0_diff = generated_waypoint_batch - future_tensor.to(device)
        # # print(xy_ped_0_diff.size())
        # # xy_ped_0_diff = xy_ped_0_diff.view(xy_ped_0_diff.size(1) * xy_ped_0_diff.size(0), 8, 2)
        # # print(xy_ped_0_diff.size())
        # # print(xy_ped_0_diff[1],generated_waypoint_batch[0],future_tensor[1])
        # # for i in range(best_of_n):
        # #     print(generated_waypoint_batch[i].size() , future_tensor.to(device).size())
        # #     xy_ped_0_diff[i,:,:] = generated_waypoint_batch[i,:,:] - future_tensor.to(device) #future_ped_batch[:,:1,:].to(device) #future_ped_batch.to(device) #future_ped_batch[:,:1,:].to(device)
        # # print('xy_ped_0_diff: ', xy_ped_0_diff)
        
        # xy_ped_0_diff_dist = torch.tensor(np.zeros(shape=(xy_ped_0_diff.size(0),xy_ped_0_diff.size(1),1)), dtype=torch.float64)
        # xy_ped_0_diff_dist[:,:,0] = torch.linalg.norm(xy_ped_0_diff, dim = 2)
        # xy_ped_0_diff_dist=xy_ped_0_diff_dist.to(device)

    
        # #STL specifications
    
        # max_dist =  0.4
        # ydist = torch.as_tensor(xy_ped_0_diff_dist).float()
        # yfdist = stlcg.Expression('yf', ydist)
        # ϕdist = yfdist > torch.as_tensor(max_dist).float()
        # ψdist = stlcg.Always(subformula=ϕdist)

        # # test_STL_ade = torch.tensor(np.ones(shape=(xy_ped_0_diff.size(0),xy_ped_0_diff.size(1),1)), dtype=torch.float64).to(device) * l2error_avg
        # # # print(STL_de[0,:,0])
        # # # print(criterion(future, generated_waypoint)/float(xy_ped_0_diff.size(0)+xy_ped_0_diff.size(1)))
        # # # STL_ade = torch.mean(STL_de,1).unsqueeze(1).to(device)
        # # # print(STL_ade.size())
        # # test_yade = torch.as_tensor(test_STL_ade).float()
        # # test_yade = stlcg.Expression('yf', test_yade)
        # # test_ϕade = test_yade <= torch.as_tensor(0.0).float()
        # # test_ϕade_dist = test_ϕdist & test_ϕade
        # # test_ψade_dist = stlcg.Always(subformula=test_ϕade_dist) 
        # # test_robustness_ade_dist = torch.relu(-test_ψade_dist.robustness((xy_ped_0_diff_dist.flip(1),0.01*test_STL_ade.flip(1)), scale=-1)).sum() 



        # # test_turn_angle = torch.tensor(np.zeros(shape=(xy_generated_waypoint.size(0),xy_generated_waypoint.size(1)-1,1)), dtype=torch.float64)
                
        # # for i in np.arange(1,xy_generated_waypoint.size(1),1):
        
        # #     test_turn_angle[:,i-1,0] = torch.atan2(xy_generated_waypoint[:,i,1],xy_generated_waypoint[:,i,0]) - torch.atan2(xy_generated_waypoint[:,i-1,1],xy_generated_waypoint[:,i-1,0])
        
        
        # # # print(turn_angle[0:5,:])
        # # turn_angle_neg = test_turn_angle - 2*torch.acos(torch.zeros(1)).item() * 2
        # # turn_angle_pos = test_turn_angle + 2*torch.acos(torch.zeros(1)).item() * 2

        # # test_turn_angle = torch.where(test_turn_angle > torch.acos(torch.zeros(1)).item() * 2, turn_angle_neg, test_turn_angle)
        # # test_turn_angle = torch.where(test_turn_angle <= -torch.acos(torch.zeros(1)).item() * 2, turn_angle_pos, test_turn_angle)
        
        # test_speed, test_turn_angle = speed_and_angle_change(xy_generated_waypoint, 0.4)

        # test_sagittal_component, test_lateral_component = velocity_in_coordinate_frame(test_speed, test_turn_angle)

        # # test_sag_safety_signal, test_lat_safety_signal = turn_safety(test_turn_angle, test_speed)

        # # test_sag_safety_signal = test_sag_safety_signal.to(device)
        # # test_lat_safety_signal = test_lat_safety_signal.to(device)
        # asym =  0.62024
        # test_sag_safety= torch.as_tensor(test_sagittal_component).float()
        # test_ysag_safety = stlcg.Expression('sag_safety', test_sag_safety)
        # lat_limit = 0.3 # ~ sin(18 deg) * 0.6  < 0.3*sqrt(9.81/1.02)*cos(18deg)
        # sag_limit = 0.6 

        # test_lat_safety= torch.as_tensor(test_lateral_component).float()
        # test_ylat_safety = stlcg.Expression('lat_safety', test_lat_safety)
        # test_safe1 = test_ysag_safety >= torch.as_tensor(0.0).float()
        # test_safe2 = test_ysag_safety <= torch.as_tensor(sag_limit).float()
        # test_safe3 = test_ylat_safety <= torch.as_tensor(lat_limit).float()
        # test_safe4 = test_ylat_safety >= torch.as_tensor(-lat_limit).float()
        # test_sag_spec = stlcg.Always(subformula= test_safe1 & test_safe2)
        # test_lat_spec = stlcg.Always(subformula=test_safe3 & test_safe4)
        # loco_safety = test_sag_spec & test_lat_spec
        # test_ψloco_safety = stlcg.And(subformula1=test_sag_spec, subformula2=test_lat_spec)

        # test_robustness_loco_safety = torch.relu(-test_ψloco_safety.robustness(((test_sagittal_component.flip(1),test_sagittal_component.flip(1)), (test_lateral_component.flip(1), test_lateral_component.flip(1))), scale=-1)).sum()
        
        # # test_robustness_loco_safety = torch.relu(-test_ψloco_safety.robustness((test_speed.flip(1), test_speed.flip(1)), scale=-1)).sum()

        # ## turn_angle stl
        # max_angle =  0.314159
        # test_yang = torch.as_tensor(test_turn_angle).float()
        # test_yturn = stlcg.Expression('yf', test_yang)
        # test_turn_ϕ2 = test_yturn > torch.as_tensor(-max_angle).float()
        # test_turn_ϕ3 = test_yturn < torch.as_tensor(max_angle).float()
        # test_turn_ϕang = test_turn_ϕ2 & test_turn_ϕ3
        # test_turn_ψang = stlcg.Always(subformula=test_turn_ϕang)

        # test_robustness_turn_ang = torch.relu(-test_turn_ψang.robustness((test_turn_angle, test_turn_angle), scale=-1)).sum()
        # ### solve for robustness ##

        # test_ψloco_safety_comb = stlcg.And(subformula1=test_ψloco_safety, subformula2=test_turn_ψang)
        
        # test_robustness_loco_safety = torch.relu(-test_ψloco_safety_comb.robustness((((test_sagittal_component.flip(1),test_sagittal_component.flip(1)),
        #                                                                                       (test_lateral_component.flip(1), test_lateral_component.flip(1))),
        #                                                                                       (test_turn_angle.flip(1), test_turn_angle.flip(1))), scale=-1)).sum()


        

        # robustness_dist = torch.relu(-ψdist.robustness(xy_ped_0_diff_dist, scale=-1)).sum()

        

        # ##########################
        # agg_features = ptraj # model.sum(x, x.size(0))
        # agg_features = agg_features.cpu().numpy()
        # agg_features = np.array(agg_features)
        # agg_features = np.reshape(agg_features,(-1, 8, 2))
      

    return all_pred





def pred_ego(N, model, device, checkpoint, hyper_params, frame, test_past_list, test_goal_list, test_waypoint_list, test_future_list):
     
    
    # model = EgoNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params["fdim"], hyper_params["zdim"], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
    model = model.double().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # all_pred = run(test_past_list[frame], test_goal_list[frame], model, device, hyper_params, best_of_n = N)
    
    # print(test_past_list[frame][0,0,:])

    all_pred = run_sum(test_past_list[frame], test_goal_list[frame], test_future_list[frame], hyper_params, model, device, 10)

    mean_all_pred = sum(all_pred)/len(all_pred)
    return mean_all_pred[:,0], mean_all_pred[:,1]


