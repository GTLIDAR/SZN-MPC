import numpy as np
import matplotlib.pyplot as plt
from math import ceil
import torch

from attrdict import AttrDict
from model.models import TrajectoryGenerator

def get_generator(checkpoint):
    args = AttrDict(checkpoint['args'])
    generator = TrajectoryGenerator(
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        embedding_dim=args.embedding_dim,
        encoder_h_dim=args.encoder_h_dim_g,
        decoder_h_dim=args.decoder_h_dim_g,
        mlp_dim=args.mlp_dim,
        num_layers=args.num_layers,
        noise_dim=args.noise_dim,
        noise_type=args.noise_type,
        noise_mix_type=args.noise_mix_type,
        pooling_type=args.pooling_type,
        pool_every_timestep=args.pool_every_timestep,
        dropout=args.dropout,
        bottleneck_dim=args.bottleneck_dim,
        neighborhood_size=args.neighborhood_size,
        grid_size=args.grid_size,
        batch_norm=args.batch_norm)
    generator.load_state_dict(checkpoint['g_state'])
    generator.cuda()
    generator.train()
    return generator

def generate_ped_trajectory_tensor(ped_traj, ped_traj_disp, init=False):
    path = "./model/checkpoint/checkpoint_with_model.pt"
    checkpoint = torch.load(path)
    generator = get_generator(checkpoint)

    num_ped = ped_traj.shape[1]
    # start_end = [(0, num_ped)] + [(num_ped, num_ped)]*(num_ped-1)
    list_ped = np.arange(num_ped+1)
    start_end = [(start, end) for start, end in zip(list_ped, list_ped[1:])]
    start_end = torch.tensor(start_end, dtype=torch.int64, device= "cuda")
    disp = generator(ped_traj, ped_traj_disp, start_end)
    displacement = torch.cumsum(disp.permute(1, 0, 2), dim=1)
    traj = displacement + ped_traj[-1].unsqueeze(1)
    trajectories = torch.concat([ped_traj,traj.permute(1, 0, 2)], dim=0)

    if not init:
        trajectories = trajectories[1:9]
        disp = disp[1:9]
    else:
        trajectories = trajectories[-8:]
        disp = disp[-8:]
    return trajectories.detach(), disp.detach()

def generate_random_trajectories(num_ped, timesteps = 8, min = 0, max = 10):
    ped = np.random.uniform(min, max, (num_ped, 2))
    disp = np.random.uniform(-0.5,0.5, (num_ped, timesteps, 2)) 
    trajectories = np.cumsum(disp, axis=1) + ped[:,np.newaxis,:]
    trajectories = trajectories.transpose(1,0,2)
    disp = disp.transpose(1,0,2)

    trajectories = torch.from_numpy(trajectories).to(dtype=torch.double, device= "cuda")
    disp = torch.from_numpy(disp).to(dtype=torch.double, device= "cuda")

    return trajectories, disp