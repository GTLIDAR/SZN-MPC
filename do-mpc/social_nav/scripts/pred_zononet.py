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

    all_generated_wayponts = torch.zeros(best_of_n, (2+(2*hyper_params["Ng"]))*(hyper_params['future_length']-1))

    ### Affine
    # all_generated_wayponts = torch.zeros(best_of_n, (2+(2*hyper_params["Ng"]))+ 6*(hyper_params['future_length']-2))
    
    
    # for _ in 1:
    _ = 0
    generated_waypoint = model.forward(ptraj, dest, device=device)
    all_generated_wayponts[_,:]=generated_waypoint
    generated_waypointnp = generated_waypoint.detach().cpu().numpy()
    all_guesses.append(generated_waypointnp)
    
    # all_guesses = generated_waypoint

    all_guesses = np.array(all_guesses)
    # all_pred = np.reshape(all_guesses,(-1, hyper_params['future_length'], 2))

    
        
      

    return generated_waypoint, all_guesses


def H_rep(centers, generator_matrices, generator_vectors, xc, yc, device):
    
    L = torch.norm(generator_matrices, dim=3)  # Calculate norm along the 3rd dimension
    
    A_poly_in = torch.empty([generator_matrices.shape[0], generator_matrices.shape[1],generator_matrices.shape[2], 2]).to(device)
    
    G = generator_matrices
    
    
    C = torch.empty(generator_matrices.shape[0], generator_matrices.shape[1], generator_matrices.shape[2], 2)
    for i in range(generator_matrices.shape[2]):
        C[:,:,i,0] = -generator_vectors[:,:,i,1]
        C[:,:,i,1] = generator_vectors[:,:,i,0]

    
    # print(generator_vectors.shape)
    # print(L.shape)
    # print(A_poly_in.shape)
    for i in range(generator_matrices.shape[2]):
        A_poly_in[:,:,i,0] =  1/L[:,:,i] * -generator_vectors[:,:,i,1]
        A_poly_in[:,:,i,1] =  1/L[:,:,i] * generator_vectors[:,:,i,0]
        

    


    A_poly = torch.cat((A_poly_in,-A_poly_in),dim=2).requires_grad_(True)
    
    centers = centers + torch.tensor([[xc,yc]]).to(device)

    ATc = torch.matmul(A_poly, centers.unsqueeze(-1))
    ATG = torch.matmul(A_poly_in, generator_matrices.transpose(2,3))
    ATG = torch.cat((ATG,ATG),dim=2)
    ones = torch.ones(generator_matrices.shape[0],generator_matrices.shape[1],generator_matrices.shape[2],1).to(device)
    b_poly = ATc + torch.matmul(torch.abs(ATG),ones).requires_grad_(True)
    
    return A_poly.squeeze(0), b_poly.squeeze(0), torch.matmul(torch.abs(ATG),ones).requires_grad_(True)


def get_center_and_genrator(generated_waypoints, ng, t):

    # Reshape the input tensor to separate examples and time steps
    k = 2*ng + 2
    # print(generated_waypoints[0,:])
    
    input_tensor = generated_waypoints.view(-1, t, ng+1,2)
    
    # Extract centers and generator vectors
    centers = input_tensor[:, :, 0] #[n,t,coord]
    
    generator_vectors = input_tensor[:, :, 1:] #[n,t,ng,coord]

    generator_matrices = generator_vectors.view(-1, t, ng, 2) #[n,t,ng,coord]
    
    return centers, generator_matrices, generator_vectors



def pred_zono(N, model, device, checkpoint, hyper_params, frame, test_past_list, test_goal_list, test_waypoint_list, test_future_list, xc, yc):
     
    
    # model = EgoNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params["fdim"], hyper_params["zdim"], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
    model = model.double().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # all_pred = run(test_past_list[frame], test_goal_list[frame], model, device, hyper_params, best_of_n = N)
    
    # print(test_past_list[frame][0,0,:])

    generated_waypoint, all_pred = run_sum(test_past_list[frame], test_goal_list[frame], test_future_list[frame], hyper_params, model, device, 1)

 
    t = 7
    ng = hyper_params["Ng"]

    c, generator_matrices, generator_vectors = get_center_and_genrator(generated_waypoint, hyper_params["Ng"], 
                                                                                         (hyper_params['future_length']-1))
    
    A_poly, b_poly, ATG = H_rep(c, generator_matrices, generator_vectors, xc, yc, device)
    ####### for affine predictions
    # centers, generator_matrices, generator_vectors = get_center_and_genrator_affine(all_pred, hyper_params["Ng"], 
                                                                                #  (hyper_params['future_length']-1), device)
    # Extract centers and generator vectors
    # centers = np.array(centers.cpu().detach().numpy())
    # generator_vectors = np.array(generator_vectors.cpu().detach().numpy()) 
    # print(A_poly.shape)
    # A_poly = A_poly.squeeze(0) #.transpose(1,2)
    # b_poly = b_poly.squeeze(0) #.transpose(1,2)
    ####### one shot predicition 
    input_tensor1 = all_pred.reshape(-1, t, ng+1,2)
    centers = input_tensor1[:, :, 0]
    generator_vectors1 = input_tensor1[:, :, 1:]
    # print(A_poly.shape)
    # print(b_poly.shape)
    # mean_all_pred = sum(all_pred)/len(all_pred)
    return centers[0,:,0], centers[0,:,1], A_poly, b_poly, generator_vectors1, generator_matrices, generator_vectors, c, ATG #np.array(A_poly.cpu().numpy()), np.array(b_poly.cpu().numpy()), generator_vectors1



def no_ped(pred_x, pred_y):
    A_poly_out = []
    b_poly_out = []

    for n in range(pred_x.shape[1]):
        g1 = np.array([0.125,0.0])
        g2 = np.array([0.0,0.125])

        G = np.c_[g1, g2, g1*0.01, g1*0.01, g1*0.01, g1*0.01, g1*0.01] #, g1*0.01, g1*0.1]

        L = np.linalg.norm(G, axis=0)
        A_poly_in = np.zeros([np.shape(G)[1],2])

        for i in range(np.shape(G)[1]-1):
            A_poly_in[i,0] =  -1/L[i] * G[1,i]
            A_poly_in[i,1] =  1/L[i] * G[0,i]
        

        A_poly = np.concatenate((A_poly_in,-A_poly_in),axis=0)
        centers = np.array([pred_x[0,n],pred_y[0,n]])

        ATc = np.dot(A_poly, centers)
        ATG = np.dot(A_poly_in, G)
        ATG = np.concatenate((ATG,ATG),axis=0)
        ones = np.ones([np.shape(G)[1],1])
        
        b_poly = ATc + np.dot(np.abs(ATG),ones)
        A_poly_out.append(A_poly)
        b_poly_out.append(b_poly)
    
    
    return np.array(A_poly_out), np.array(b_poly_out), G, np.dot(np.abs(ATG),ones)



def mink_sum(ped_x, ped_y, generator_ego):       
    g1 = np.array([0.125,0.0])
    g2 = np.array([0.0,0.125])

    G_ped = np.c_[g1, g2]#, g1*0.01, g1*0.01, g1*0.01]
    G=np.concatenate((G_ped, generator_ego.T),axis=1)
    L = np.linalg.norm(G, axis=0)
    A_poly_in = np.zeros([np.shape(G)[1],2])

    for i in range(np.shape(G)[1]):
        A_poly_in[i,0] =  -1/L[i] * G[1,i]
        A_poly_in[i,1] =  1/L[i] * G[0,i]

    

    A_poly = np.concatenate((A_poly_in,-A_poly_in),axis=0)
    centers = np.array([ped_x,ped_y])

    ATc = np.dot(A_poly, centers)
    ATG = np.dot(A_poly_in, G)
    ATG = np.concatenate((ATG,ATG),axis=0)
    ones = np.ones([np.shape(G)[1],1])
    
    b_poly = ATc + np.dot(np.abs(ATG),ones)
    
    
    
    return A_poly, b_poly, G


def ped_pred_zono(N, model, device, checkpoint, hyper_params, frame, test_past_list, test_future_list):
     
    
    # model = EgoNet(hyper_params["enc_past_size"], hyper_params["enc_dest_size"], hyper_params["enc_latent_size"], hyper_params["dec_size"], hyper_params["predictor_hidden_size"], hyper_params["fdim"], hyper_params["zdim"], hyper_params["sigma"], hyper_params["past_length"], hyper_params["future_length"], args.verbose)
    model = model.double().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # all_pred = run(test_past_list[frame], test_goal_list[frame], model, device, hyper_params, best_of_n = N)
    

    past_tensor = test_past_list[frame] - test_past_list[frame][:,0,:].unsqueeze(1)
    future_tensor = test_past_list[frame] - test_past_list[frame][:,0,:].unsqueeze(1)
    all_pred = ped_run_sum(past_tensor, future_tensor, model, device, 1)

    # print(all_pred.shape)
    t = 7
    ng = hyper_params["Ng"]

    # c, generator_matrices, generator_vectors = get_center_and_genrator(generated_waypoint, hyper_params["Ng"], 
    #                                                                                      (hyper_params['future_length']-1))
    
    # A_poly, b_poly, ATG = H_rep(c, generator_matrices, generator_vectors, xc, yc, device)
    ####### for affine predictions
    # centers, generator_matrices, generator_vectors = get_center_and_genrator_affine(all_pred, hyper_params["Ng"], 
                                                                                #  (hyper_params['future_length']-1), device)
    # Extract centers and generator vectors
    # centers = np.array(centers.cpu().detach().numpy())
    # generator_vectors = np.array(generator_vectors.cpu().detach().numpy()) 
    # print(A_poly.shape)
    # A_poly = A_poly.squeeze(0) #.transpose(1,2)
    # b_poly = b_poly.squeeze(0) #.transpose(1,2)
    ####### one shot predicition 
    input_tensor1 = all_pred.reshape(-1, t, ng+1,2)
    
    centers = input_tensor1[:, :, 0] + test_past_list[frame][:,0,:].numpy()[:,np.newaxis,:]

    generator_vectors1 = input_tensor1[:, :, 1:]
    # print(A_poly.shape)
    # print(b_poly.shape)
    # mean_all_pred = sum(all_pred)/len(all_pred)
    return centers, generator_vectors1 #np.array(A_poly.cpu().numpy()), np.array(b_poly.cpu().numpy()), generator_vectors1

def ped_run_sum(ped_traj_tensor, future_tensor, model, device, best_of_n = 1):
    model.eval()
    assert best_of_n >= 1 and type(best_of_n) == int
    
    with torch.no_grad():
                
        initial_pos = ped_traj_tensor[:,-1,:].double()

        xx = ped_traj_tensor.view(-1, ped_traj_tensor.shape[1]*ped_traj_tensor.shape[2]).double()
        
        x = xx.to(device)
        
        dest_recon = model.forward(x, initial_pos, device=device)
        generated_waypoint = model.predict(x, dest_recon, initial_pos)

        all_guesses = []
        generated_waypointnp = generated_waypoint.cpu().numpy()
        all_guesses.append(generated_waypointnp)

       
        
        # all_guesses = generated_waypoint
        all_guesses = np.array(all_guesses)

      

    return generated_waypointnp