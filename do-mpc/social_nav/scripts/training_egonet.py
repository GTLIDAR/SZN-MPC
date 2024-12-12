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


parser = argparse.ArgumentParser(description='PECNet')

parser.add_argument('--num_workers', '-nw', type=int, default=0)
parser.add_argument('--gpu_index', '-gi', type=int, default=0)
parser.add_argument('--config_filename', '-cfn', type=str, default='optimal_eth_ucy.yaml')
parser.add_argument('--save_file', '-sf', type=str, default='PECNET_social_model_eth.pt')
parser.add_argument('--verbose', '-v', action='store_true')

args = parser.parse_args()

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)
print(device)

with open("../config/" + args.config_filename, 'r') as file:
    try:
        hyper_params = yaml.load(file, Loader = yaml.FullLoader)
    except:
        hyper_params = yaml.load(file)
file.close()
print(hyper_params)

def train(past_list, goal_list, waypoint_list):

    model.train()
    train_loss = 0
    total_rcl, total_kld, total_adl = 0, 0, 0
    criterion = nn.MSELoss()

    for i, (ped_traj_tensor, goal_tensor, robot_tensor) in enumerate(zip(past_list, goal_list, waypoint_list)):
        #     traj, mask, initial_pos = torch.DoubleTensor(traj).to(device), torch.DoubleTensor(mask).to(device), torch.DoubleTensor(initial_pos).to(device)
        x = ped_traj_tensor
        y = goal_tensor
        
        # print(i)
        # print(x.size())
        # print(x)
        # print(initial_pos.size())
        # print(initial_pos)
        # print(i)

        x = x.contiguous().view(-1, x.shape[1]*x.shape[2]) # (x,y,x,y ... )
        # print(x.size())
        x = x.to(device)
        dest = y[:, -1, :].to(device)
        # print(dest.size())
        # print(robot_tensor.size())
        future = robot_tensor.contiguous().view(y.size(0),-1).to(device)
        # print(future.size())
        # print(dest)

        #def forward(self, x, dest, waypoint = None, device=torch.device('cpu')):


        generated_waypoint, mu, var, interpolated_future = model.forward(x, dest, waypoint=future, device=device)
        
        gen_final_d = generated_waypoint[:, 6:8]
        
        # print(future[0,:])
        # print(future[0, 2*hyper_params["future_length"]-2:2*hyper_params["future_length"]])
        # print(future[:, -1])
        optimizer.zero_grad()
        rcl, kld, adl = calculate_loss(future[:, 6:8], gen_final_d, mu, var, criterion, future, generated_waypoint)
        loss = 0.5*rcl + kld*hyper_params["kld_reg"] + adl*hyper_params["adl_reg"]*100
        loss.backward()
        train_loss += loss.item()
        total_rcl += rcl.item()
        total_kld += kld.item()
        total_adl += adl.item()
        optimizer.step()

    return train_loss, total_rcl, total_kld, total_adl

def test(past_list, goal_list, waypoint_list, best_of_n = 1):
    model.eval()
    assert best_of_n >= 1 and type(best_of_n) == int
    test_loss = 0

    with torch.no_grad():
        total_seen = 0
        total_ade = 0
        total_fde = 0
        for i, (ped_traj_tensor, goal_tensor, robot_tensor) in enumerate(zip(past_list, goal_list, waypoint_list)):
            total_seen += len(ped_traj_tensor)
            
            x = ped_traj_tensor
            y = goal_tensor
            y = y.cpu().numpy()
            

            # reshape the data
            x = x.view(-1, x.shape[1]*x.shape[2])
            x = x.to(device)

            dest = torch.from_numpy(y[:, -1, :]).to(device)

            waypoints = robot_tensor.contiguous().view(goal_tensor.size(0),-1).cpu().numpy()
            # print(dest)
            # dest = y[:, -1, :]
            
            all_l2_errors_dest = []
            all_guesses = []
            for _ in range(best_of_n):

                generated_waypoint = model.forward(x, dest, device=device)
                generated_waypoint = generated_waypoint.cpu().numpy()
                all_guesses.append(generated_waypoint)

                l2error_sample = np.linalg.norm(torch.from_numpy(generated_waypoint) - waypoints, axis = 1)
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

            # print(predicted_future)
            # ADE error
            l2error_overall = np.mean(np.linalg.norm(waypoints_xy - predicted_future, axis = 2))

            l2error_overall /= hyper_params["data_scale"]
            l2error_dest /= hyper_params["data_scale"]
            l2error_avg_dest /= hyper_params["data_scale"]

            total_ade += (l2error_overall*len(x))
            total_fde += (l2error_dest*len(x))

            # print('Test time error in destination best: {:0.3f} and mean: {:0.3f}'.format(l2error_dest, l2error_avg_dest))
            # print('Test time error overall (ADE) best: {:0.3f}'.format(l2error_overall))

    return (total_ade/total_seen), (total_fde/total_seen), l2error_avg_dest

model = EgoNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params["fdim"], hyper_params["zdim"], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
model = model.double().to(device)
optimizer = optim.Adam(model.parameters(), lr=  hyper_params["learning_rate"])

# train_dataset = SocialDatasetETHUCY(set_name="zara1", set_type='train', b_size=hyper_params["train_b_size"], t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"])
# test_dataset = SocialDatasetETHUCY(set_name="zara1", set_type='test', b_size=512, t_tresh=hyper_params["time_thresh"], d_tresh=hyper_params["dist_thresh"])




best_test_loss = 50 # start saving after this threshold
best_endpoint_loss = 50
N = hyper_params["n_values"]
robot_batch = 300

train_past_list, train_goal_list, train_waypoint_list = get_data('/home/ashamsah3/Human-Path-Prediction/PECNet/datatext/students001.txt', robot_batch, hyper_params["past_length"], hyper_params["future_length"])
test_past_list, test_goal_list, test_waypoint_list = get_data('/home/ashamsah3/Human-Path-Prediction/PECNet/datatext/students003_val.txt', robot_batch, hyper_params["past_length"], hyper_params["future_length"])

# shift origin and scale data

# for i, (train_past_list, train_goal_list, train_waypoint_list) in enumerate(zip(train_past_list, train_goal_list, train_waypoint_list)):
#     train_past_list -= train_waypoint_list[0,0,0]
#     # train_past_list *= hyper_params["data_scale"]
#     train_goal_list -= train_waypoint_list[0,0,0]
#     # train_goal_list *= hyper_params["data_scale"]
#     train_waypoint_list -= train_waypoint_list[0,0,0]
#     # train_waypoint_list *= hyper_params["data_scale"]

# for i, (test_past_list, test_goal_list, test_waypoint_list) in enumerate(zip(test_past_list, test_goal_list, test_waypoint_list)):
#     test_past_list -= test_waypoint_list[:,:1,:]
#     test_past_list *= hyper_params["data_scale"]
#     test_goal_list -= test_waypoint_list[:,:1,:]
#     test_goal_list *= hyper_params["data_scale"]
#     test_waypoint_list -= test_waypoint_list[:,:1,:]
#     test_waypoint_list *= hyper_params["data_scale"]



print('##################  Done loading data ##################' )
for e in range(hyper_params['num_epochs']):
    train_loss, rcl, kld, adl = train(train_past_list, train_goal_list, train_waypoint_list)
    test_loss, final_point_loss_best, final_point_loss_avg = test(test_past_list, test_goal_list, test_waypoint_list, best_of_n = N)

    # print()

    if best_test_loss > test_loss:
        print("Epoch: ", e)
        print('################## BEST PERFORMANCE {:0.2f} ########'.format(test_loss))
        best_test_loss = test_loss
        if best_test_loss < 10.25:
            save_path = '../saved_models/' + args.save_file
            torch.save({
                        'hyper_params': hyper_params,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        }, save_path)
            print("Saved model to:\n{}".format(save_path))

    if final_point_loss_best < best_endpoint_loss:
        best_endpoint_loss = final_point_loss_best
    print("Epoch: ", e)
    print("Train Loss", train_loss)
    print("RCL", rcl)
    print("KLD", kld)
    print("ADL", adl)
    print("Test ADE", test_loss)
    print("Test Average FDE (Across  all samples)", final_point_loss_avg)
    print("Test Min FDE", final_point_loss_best)
    print("Test Best ADE Loss So Far (N = {})".format(N), best_test_loss)
    print("Test Best Min FDE (N = {})".format(N), best_endpoint_loss)
