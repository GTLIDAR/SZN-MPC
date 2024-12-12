import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from torch.utils.data import DataLoader
import argparse
sys.path.append("../eth_ucy/")
from social_eth_ucy_utils import *

sys.path.append("../utils/")
import yaml
from models import *
import numpy as np
from read_txt_dataset import *
import time

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PECNet')

parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--load_file', '-lf', default="run7.pt")
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


checkpoint = torch.load('../saved_models/{}'.format(args.load_file), map_location=device)
hyper_params = checkpoint["hyper_params"]

print(hyper_params)

def test(ped_traj_tensor, goal_tensor, robot_tensor, model, best_of_n = 1):
    model.eval()
    assert best_of_n >= 1 and type(best_of_n) == int
    test_loss = 0

    with torch.no_grad():
        total_seen = 0
        total_ade = 0
        total_fde = 0
        x = ped_traj_tensor
        y = goal_tensor
        y = y.cpu().numpy()
        
        total_seen += len(ped_traj_tensor)
        # reshape the data
        x = x.view(-1, x.shape[1]*x.shape[2])
        x = x.to(device)

        dest = torch.from_numpy(y[:, -1, :]).to(device)
        
        waypoints = robot_tensor[:,].contiguous().view(goal_tensor.size(0),-1).cpu().numpy()
        future = robot_tensor[:,].contiguous().view(goal_tensor.size(0),-1)
        future = future[:,10:12].cpu().numpy()
        # print(future.size)
        # dest = y[:, -1, :]
        
        all_l2_errors_dest = []
        all_guesses = []
        
        for _ in range(best_of_n):

            generated_waypoint = model.forward(x, dest, device=device)
            
            generated_waypoint2 = generated_waypoint[:,10:12].cpu().numpy()
            generated_waypoint = generated_waypoint.cpu().numpy()
           
            all_guesses.append(generated_waypoint)
            # print(generated_waypoint.size)
            l2error_sample = np.linalg.norm(torch.from_numpy(generated_waypoint) - waypoints, axis = 1)
            # l2error_sample = np.linalg.norm(torch.from_numpy(generated_waypoint2) - future, axis = 1)
            all_l2_errors_dest.append(l2error_sample)

        all_l2_errors_dest = np.array(all_l2_errors_dest)
        all_guesses = np.array(all_guesses)
        
        # average error
        l2error_avg_dest = np.mean(all_l2_errors_dest)

        # choosing the best guess
        indices = np.argmin(all_l2_errors_dest, axis = 0)

        best_guess_dest = all_guesses[indices,np.arange(x.shape[0]),  :]

        # taking the minimum error out of all guess
        l2error_dest = np.mean(np.min(all_l2_errors_dest, axis = 0))

        best_guess_dest = torch.DoubleTensor(best_guess_dest).to(device)

        # using the best guess for interpolation
        interpolated_future = best_guess_dest #model.predict(x, best_guess_dest, mask, initial_pos)
        # interpolated_future = interpolated_future.cpu().numpy()
        best_guess_dest = best_guess_dest.cpu().numpy()

        # final overall prediction
        # predicted_future = np.concatenate((interpolated_future, best_guess_dest), axis = 1)
        predicted_future = np.reshape(best_guess_dest, (-1, hyper_params['future_length'], 2)) # making sure
        waypoints_xy = np.reshape(waypoints, (-1, hyper_params['future_length'], 2))

        all_pred = np.reshape(all_guesses,(-1, hyper_params['future_length'], 2))
        # print(predicted_future)
        # ADE error
        l2error_overall = np.mean(np.linalg.norm(waypoints_xy - predicted_future, axis = 2))

        # l2error_overall /= hyper_params["data_scale"]
        # l2error_dest /= hyper_params["data_scale"]
        # l2error_avg_dest /= hyper_params["data_scale"]

        total_ade += (l2error_overall*len(x))
        total_fde += (l2error_dest*len(x))

            # print('Test time error in destination best: {:0.3f} and mean: {:0.3f}'.format(l2error_dest, l2error_avg_dest))
            # print('Test time error overall (ADE) best: {:0.3f}'.format(l2error_overall))

    return (total_ade/total_seen), (total_fde/total_seen), l2error_avg_dest, predicted_future, all_pred




def main():
     N = 2 #args.num_trajectories
    #  print('number of samples N: ', N)
     model = EgoNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params["fdim"], hyper_params["zdim"], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
     model = model.double().to(device)
     model.load_state_dict(checkpoint["model_state_dict"])

     test_past_list, test_goal_list, test_waypoint_list = get_data('/home/ashamsah3/social_planner/EgoNet/datatext/students001.txt', 1, hyper_params["past_length"], hyper_params["future_length"])
     num_samples = 1
     
     for i in range(num_samples):
        t = time.time()
        test_loss, final_point_loss_best, final_point_loss_avg, social_traj, all_pred = test(test_past_list[0], test_goal_list[0], test_waypoint_list[0], model, best_of_n = N)
        elapsed = time.time() -t
        print('time to predict: ', elapsed)
        social_traj = social_traj
        test_loss += test_loss
     
     
     mean_traj_0 = social_traj/num_samples
     mean_traj = sum(mean_traj_0)/len(mean_traj_0)
     mean_all_pred = sum(all_pred)/len(all_pred)


     print('ADE:', test_loss/num_samples)

     fig = plt.figure()
     ax1 = fig.add_subplot(1,1,1)
     shortest= np.array([[test_waypoint_list[0][0][0,0], test_waypoint_list[0][0][0,1]],[test_goal_list[0][0,0,0],test_goal_list[0][0,0,1]]])
     ax1.scatter(mean_traj[:,0],mean_traj[:,1], label = 'mean of best guess*', color = 'red', s = 20, marker = "+")
     ax1.plot(shortest[:,0],shortest[:,1], label = 'shortest line', color = 'blue', alpha = 0.2)
     ax1.scatter(test_waypoint_list[0][0][:,0], test_waypoint_list[0][0][:,1], label = 'ground truth', color = 'red', s = 20)
     ax1.scatter(test_goal_list[0][0,0,0],test_goal_list[0][0,0,1], label = 'goal', color = 'green', marker="X", s = 150)
     
        

    #  for i in range(len(all_pred)):
    #      ax1.plot(all_pred[i,:,0],all_pred[i,:,1], color = 'blue', alpha = 0.01)

     for i in range(test_goal_list[0].size(0)):
        ax1.scatter(test_past_list[0][i,0,0],test_past_list[0][i,0,1], color = 'green')
        ax1.plot(test_past_list[0][i,:,0],test_past_list[0][i,:,1], color = 'green', linestyle = '--')

        # ax1.scatter(mean_traj_0[0][i,0,0],mean_traj_0[0][i,0,1], color = 'green')
        # ax1.plot(mean_traj_0[i,:,0],mean_traj_0[i,:,1], color = 'red', alpha = 0.1)
    
     ax1.plot(mean_all_pred[:,0], mean_all_pred[:,1], color = 'black', label = 'mean of all guesses')# s = 20, marker = "+")
     ax1.legend()

     plt.show()



     


main()
