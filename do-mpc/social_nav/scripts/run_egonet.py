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
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle

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

def run(ped_traj_tensor, goal_tensor, model, best_of_n = 1):
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




def main():
     N = 5 #args.num_trajectories
    #  print('number of samples N: ', N)
     model = EgoNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params["fdim"], hyper_params["zdim"], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
     model = model.double().to(device)
     model.load_state_dict(checkpoint["model_state_dict"])

     test_past_list, test_goal_list, test_waypoint_list = get_seq_data('/home/ashamsah3/social_planner/EgoNet/datatext/students003_val.txt', 1, hyper_params["past_length"], hyper_params["future_length"])
     num_samples = len(test_past_list)
     fig = plt.figure()
     ax1 = fig.add_subplot(1,1,1)
     border = Rectangle((-10,-10), 25, 25, fc = "None", ec="black" )
    #  circle1 = Circle((0, 0), 3, alpha=0.5)
     plt.draw()
     plt.pause(1)
     
     
     for i in range(num_samples):
        t = time.time()
        # print(i)
        all_pred = run(test_past_list[i], test_goal_list[i], model, best_of_n = N)
        elapsed = time.time()-t
        print('time to predict: ', elapsed)
        mean_all_pred = sum(all_pred)/len(all_pred)
    
        
        shortest= np.array([[test_waypoint_list[i][0][0,0], test_waypoint_list[i][0][0,1]],[test_goal_list[i][0,0,0],test_goal_list[i][0,0,1]]])
        ax1.plot(shortest[:,0],shortest[:,1], label = 'shortest line', color = 'blue', alpha = 0.2)
        ax1.plot(test_waypoint_list[i][0][:,0], test_waypoint_list[i][0][:,1], label = 'ground truth', color = 'red')
        ax1.scatter(test_waypoint_list[i][0][0,0], test_waypoint_list[i][0][0,1], label = 'ground truth', color = 'red')
        ax1.scatter(test_goal_list[i][0,0,0],test_goal_list[i][0,0,1], label = 'goal', color = 'green', marker="X", s = 150)

        for j in range(test_goal_list[i].size(0)):
            ax1.scatter(test_past_list[i][j,0,0],test_past_list[i][j,0,1], color = 'green')
            ax1.plot(test_past_list[i][j,:,0],test_past_list[i][j,:,1], color = 'green', linestyle = '--')

        # ax1.plot(mean_all_pred[:,0], mean_all_pred[:,1], color = 'black', label = 'mean of all guesses')# s = 20, marker = "+")\
        for k in range(len(all_pred)):
            ax1.plot(all_pred[k,:,0],all_pred[k,:,1], color = 'black', alpha = 0.05)
        # ax1.add_patch(circle1)
        ax1.add_patch(border)
        # ax1.legend()
        plt.draw()
        plt.pause(0.3)
        plt.cla()

       



     


main()
