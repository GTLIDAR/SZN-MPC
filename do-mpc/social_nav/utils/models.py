import torch
import torch.nn as nn
from torch.autograd import Variable
import random
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import pdb
from torch.nn import functional as F
from torch.distributions.normal import Normal
import math
import numpy as np
import yaml

'''MLP model'''
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        dims.append(output_dim)
        self.layers = nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.sigmoid = nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = self.sigmoid(x)
        return x

class PECNet(nn.Module):

    def __init__(self, enc_past_size, enc_dest_size, enc_latent_size, dec_size, predictor_size, pdim, ddim, fdim, zdim, sigma, past_length, future_length, Ng, verbose):
        '''
        Args:
            size parameters: Dimension sizes
            nonlocal_pools: Number of nonlocal pooling operations to be performed
            sigma: Standard deviation used for sampling N(0, sigma)
            past_length: Length of past history (number of timesteps)
            future_length: Length of future trajectory to be predicted
        '''
        super(PECNet, self).__init__()

        self.zdim = zdim
        self.sigma = sigma

        # takes in the past
        self.encoder_past = MLP(input_dim = past_length*2, output_dim = fdim, hidden_size=enc_past_size)

        self.encoder_dest = MLP(input_dim = 2, output_dim = fdim, hidden_size=enc_dest_size)

        self.encoder_latent = MLP(input_dim = 2*fdim, output_dim = 2*zdim, hidden_size=enc_latent_size)

        self.decoder = MLP(input_dim = fdim + zdim, output_dim = 2, hidden_size=dec_size)

        # self.predictor = MLP(input_dim = 2*fdim + 2, output_dim = 2*(future_length-1), hidden_size=predictor_size)
        self.predictor = MLP(input_dim = 2*fdim + 2, output_dim = (2+(2*Ng))*(future_length-1), hidden_size=predictor_size)

        architecture = lambda net: [l.in_features for l in net.layers] + [net.layers[-1].out_features]

        if verbose:
            print("Past Encoder architecture : {}".format(architecture(self.encoder_past)))
            print("Dest Encoder architecture : {}".format(architecture(self.encoder_dest)))
            print("Latent Encoder architecture : {}".format(architecture(self.encoder_latent)))
            print("Decoder architecture : {}".format(architecture(self.decoder)))
            print("Predictor architecture : {}".format(architecture(self.predictor)))

            print("Non Local Theta architecture : {}".format(architecture(self.non_local_theta)))
            print("Non Local Phi architecture : {}".format(architecture(self.non_local_phi)))
            print("Non Local g architecture : {}".format(architecture(self.non_local_g)))

    
    def forward(self, x, initial_pos, dest = None, device=torch.device('cpu')):

        # provide destination iff training
        # assert model.training
        assert self.training ^ (dest is None)

        # encode
        # print('x:', x.size())
        ftraj = self.encoder_past(x)
        # print('ftraj: ',ftraj.size())
        if not self.training:
            z = torch.Tensor(x.size(0), self.zdim)
            z.normal_(0, self.sigma)

        else:
            # during training, use the destination to produce generated_dest and use it again to predict final future points

            # CVAE code
            # print('dest  ', dest.size())
            dest_features = self.encoder_dest(dest)
            # print('dest features: ', dest_features.size())
            features = torch.cat((ftraj, dest_features), dim = 1)
            # print('features:',features.size())
            latent =  self.encoder_latent(features)

            mu = latent[:, 0:self.zdim] # 2-d array
            logvar = latent[:, self.zdim:] # 2-d array

            var = logvar.mul(0.5).exp_()
            eps = torch.DoubleTensor(var.size()).normal_()
            eps = eps.to(device)
            z = eps.mul(var).add_(mu)

        z = z.double().to(device)
        decoder_input = torch.cat((ftraj, z), dim = 1)
        generated_dest = self.decoder(decoder_input)

        if self.training:
            # prediction in training, no best selection
            generated_dest_features = self.encoder_dest(generated_dest)

            prediction_features = torch.cat((ftraj, generated_dest_features, initial_pos), dim = 1)

            pred_future = self.predictor(prediction_features)
            return generated_dest, mu, logvar, pred_future

        return generated_dest

    # separated for forward to let choose the best destination
    def predict(self, past, generated_dest, initial_pos):
        ftraj = self.encoder_past(past)
        generated_dest_features = self.encoder_dest(generated_dest)
        prediction_features = torch.cat((ftraj, generated_dest_features, initial_pos.to(ftraj.device)), dim = 1)

        interpolated_future = self.predictor(prediction_features)
        return interpolated_future



class EgoNet(nn.Module):

    def __init__(self, enc_past_size, enc_dest_size, enc_latent_size, dec_size, predictor_size, fdim, zdim, sigma, past_length, future_length, verbose):
        '''
        Args:
            size parameters: Dimension sizes
            nonlocal_pools: Number of nonlocal pooling operations to be performed
            sigma: Standard deviation used for sampling N(0, sigma)
            past_length: Length of past history (number of timesteps)
            future_length: Length of future trajectory to be predicted
        '''
        super(EgoNet, self).__init__()

        self.zdim = zdim
        self.sigma = sigma

        # takes in the past
        self.encoder_past = MLP(input_dim = past_length*2, output_dim = fdim, hidden_size=enc_past_size)

        self.encoder_dest = MLP(input_dim = 2, output_dim = fdim, hidden_size=enc_dest_size)

        self.encoder_waypoint = MLP(input_dim = 2*future_length, output_dim = fdim, hidden_size=enc_dest_size)

        self.encoder_latent = MLP(input_dim = 3*fdim, output_dim = 2*zdim, hidden_size=enc_latent_size)

        self.decoder = MLP(input_dim = 2*fdim + zdim, output_dim = 2*future_length, hidden_size=dec_size)


        self.predictor = MLP(input_dim = 3*fdim , output_dim = 2*(future_length), hidden_size=predictor_size)

        architecture = lambda net: [l.in_features for l in net.layers] + [net.layers[-1].out_features]

        if verbose:
            print("Past Encoder architecture : {}".format(architecture(self.encoder_past)))
            print("Dest Encoder architecture : {}".format(architecture(self.encoder_dest)))
            print("Latent Encoder architecture : {}".format(architecture(self.encoder_latent)))
            print("Decoder architecture : {}".format(architecture(self.decoder)))
            print("Predictor architecture : {}".format(architecture(self.predictor)))



    def forward(self, x, dest, waypoint = None, device=torch.device('cpu')):

        # provide destination iff training
        # assert model.training
        assert self.training ^ (waypoint is None)
        
        # encode
        ptraj = self.encoder_past(x)

        dest_enc = self.encoder_dest(dest)

        env_feat = torch.cat((ptraj, dest_enc), dim = 1)

        if not self.training:
            z = torch.Tensor(env_feat.size(0), self.zdim)
            z.normal_(0, self.sigma)

        else:
            # during training, use the destination to produce generated_dest and use it again to predict final future points

            # CVAE code
            ftraj = self.encoder_waypoint(waypoint)
            
            global_feat = torch.cat((env_feat, ftraj), dim = 1)
            latent =  self.encoder_latent(global_feat)

            mu = latent[:, 0:self.zdim] # 2-d array
            logvar = latent[:, self.zdim:] # 2-d array

            var = logvar.mul(0.5).exp_()
            eps = torch.DoubleTensor(var.size()).normal_()
            eps = eps.to(device)
            z = eps.mul(var).add_(mu)

        z = z.double().to(device)
        decoder_input = torch.cat((env_feat, z), dim = 1)
        generated_waypoint = self.decoder(decoder_input)

        if self.training:
            generated_waypoint_features = self.encoder_waypoint(generated_waypoint)

            prediction_features = torch.cat((env_feat, generated_waypoint_features), dim = 1)
            

            pred_future = self.predictor(prediction_features)
            return generated_waypoint , mu, logvar, pred_future

        return generated_waypoint


class EgoNet_sum(nn.Module):

    def __init__(self, enc_past_size, enc_dest_size, enc_latent_size, dec_size, predictor_size, pdim, ddim, fdim, zdim, sigma, past_length, future_length, verbose):
        '''
        Args:
            size parameters: Dimension sizes
            nonlocal_pools: Number of nonlocal pooling operations to be performed
            sigma: Standard deviation used for sampling N(0, sigma)
            past_length: Length of past history (number of timesteps)
            future_length: Length of future trajectory to be predicted
        '''
        super(EgoNet_sum, self).__init__()

        self.zdim = zdim
        self.sigma = sigma

        # takes in the past

        self.encoder_past_sum = MLP(input_dim = past_length*2, output_dim = pdim, hidden_size=enc_past_size)

        self.encoder_past = MLP(input_dim = past_length*2, output_dim = pdim, hidden_size=enc_past_size)

        self.encoder_dest = MLP(input_dim = 2, output_dim = ddim, hidden_size=enc_dest_size)

        self.encoder_waypoint = MLP(input_dim = 2*future_length, output_dim = fdim, hidden_size=enc_past_size)

        self.encoder_latent = MLP(input_dim = pdim + ddim + fdim, output_dim = 2*zdim, hidden_size=enc_latent_size)

        self.decoder = MLP(input_dim = pdim + ddim + zdim, output_dim = 2*future_length, hidden_size=dec_size)


        self.predictor = MLP(input_dim = 3*fdim , output_dim = 2*(future_length), hidden_size=predictor_size)

        architecture = lambda net: [l.in_features for l in net.layers] + [net.layers[-1].out_features]

        if verbose:
            print("Past Encoder architecture : {}".format(architecture(self.encoder_past)))
            print("Dest Encoder architecture : {}".format(architecture(self.encoder_dest)))
            print("Latent Encoder architecture : {}".format(architecture(self.encoder_latent)))
            print("Decoder architecture : {}".format(architecture(self.decoder)))
            print("Predictor architecture : {}".format(architecture(self.predictor)))


    def sum(self, x, ped_size):

        for i in np.arange(0,ped_size,1):
        #    print(x[i].size())
           ptraj = self.encoder_past_sum(x[i,:])
        #    print('ptraj', ptraj.size())
        #    print(ptraj.get_device())
           if i == 0:
               sum_ptraj =  ptraj
           else:
               sum_ptraj = sum_ptraj + ptraj

        return ptraj.view(1,sum_ptraj.size(0))

    def forward(self, ptraj, dest, waypoint = None, device=torch.device('cpu')):

        # provide destination iff training
        # assert model.training
        assert self.training ^ (waypoint is None)
        
        # encode
        # for i in np.arange(0,ped_size,1):
        # #    print(x[i].size())
        #    ptraj = self.encoder_past_sum(x[i,:])
        #    if i == 0:
        #        sum_ptraj =  ptraj
        #    else:
        #        sum_ptraj = sum_ptraj + ptraj



        # ptraj = self.encoder_past(x)
        # print('ptraj:', ptraj.size())
        # print('dest:', dest.size())
        # print(ptraj.get_device())
        

        dest_enc = self.encoder_dest(dest)
        # print(dest_enc.get_device())
        # print('dest_enc:', dest_enc.size())
        # env_feat = torch.cat((ptraj.view(1,8), dest_enc), dim = 1)
        env_feat = torch.cat((ptraj, dest_enc), dim = 1)

        if not self.training:
            
            z = torch.Tensor(env_feat.size(0), self.zdim)
            z.normal_(0, self.sigma)
        else:
            # during training, use the destination to produce generated_dest and use it again to predict final future points

            # CVAE code
            ftraj = self.encoder_waypoint(waypoint)
            
            global_feat = torch.cat((env_feat, ftraj), dim = 1)
            latent =  self.encoder_latent(global_feat)

            mu = latent[:, 0:self.zdim] # 2-d array
            logvar = latent[:, self.zdim:] # 2-d array
            # print(logvar.size())

            var = logvar.mul(0.5).exp_()
            eps = torch.DoubleTensor(var.size()).normal_()
            eps = eps.to(device)
            z = eps.mul(var).add_(mu)
            # print(z.size())

        z = z.double().to(device)
        # print('here', env_feat.size())
        decoder_input = torch.cat((env_feat, z), dim = 1)
        # print(decoder_input.size())
        generated_waypoint = self.decoder(decoder_input)

        if self.training:
            generated_waypoint_features = self.encoder_waypoint(generated_waypoint)

            # prediction_features = torch.cat((env_feat, generated_waypoint_features), dim = 1)
            

            # pred_future = self.predictor(prediction_features)
            return generated_waypoint , mu, logvar#, pred_future

        return generated_waypoint

class ZonoNet_sum(nn.Module):

    def __init__(self, enc_past_size, enc_dest_size, enc_latent_size, dec_size, predictor_size, pdim, ddim, fdim, zdim, sigma, past_length, future_length, Ng, verbose):
        '''
        Args:
            size parameters: Dimension sizes
            nonlocal_pools: Number of nonlocal pooling operations to be performed
            sigma: Standard deviation used for sampling N(0, sigma)
            past_length: Length of past history (number of timesteps)
            future_length: Length of future trajectory to be predicted
        '''
        super(ZonoNet_sum, self).__init__()

        self.zdim = zdim
        self.sigma = sigma

        # takes in the past

        self.encoder_past_sum = MLP(input_dim = past_length*2, output_dim = pdim, hidden_size=enc_past_size)

        self.encoder_past = MLP(input_dim = past_length*2, output_dim = pdim, hidden_size=enc_past_size)

        self.encoder_dest = MLP(input_dim = 2, output_dim = ddim, hidden_size=enc_dest_size)

        self.encoder_waypoint = MLP(input_dim = 2*future_length, output_dim = fdim, hidden_size=enc_past_size)

        self.encoder_latent = MLP(input_dim = pdim + ddim + fdim, output_dim = 2*zdim, hidden_size=enc_latent_size)

        self.decoder = MLP(input_dim = pdim + ddim + zdim, output_dim = (2+(2*Ng))*(future_length-1), hidden_size=dec_size) ##Ng number of generators
        # self.decoder = MLP(input_dim = pdim + ddim + zdim, output_dim = (2+(2*Ng)) + 6*(future_length-2), hidden_size=dec_size) 


        self.predictor = MLP(input_dim = 3*fdim , output_dim = 2*(future_length), hidden_size=predictor_size)

        architecture = lambda net: [l.in_features for l in net.layers] + [net.layers[-1].out_features]

        if verbose:
            print("Past Encoder architecture : {}".format(architecture(self.encoder_past)))
            print("Dest Encoder architecture : {}".format(architecture(self.encoder_dest)))
            print("Latent Encoder architecture : {}".format(architecture(self.encoder_latent)))
            print("Decoder architecture : {}".format(architecture(self.decoder)))
            print("Predictor architecture : {}".format(architecture(self.predictor)))


    def sum(self, x, ped_size):

        for i in np.arange(0,ped_size,1):
        #    print(x[i].size())
           ptraj = self.encoder_past_sum(x[i,:])
        #    print('ptraj', ptraj.size())
        #    print(ptraj.get_device())
           if i == 0:
               sum_ptraj =  ptraj
           else:
               sum_ptraj = sum_ptraj + ptraj

        return ptraj.view(1,sum_ptraj.size(0))

    def forward(self, ptraj, dest, waypoint = None, device=torch.device('cpu')):

        # provide destination iff training
        # assert model.training
        assert self.training ^ (waypoint is None)
        
        # encode
        # for i in np.arange(0,ped_size,1):
        # #    print(x[i].size())
        #    ptraj = self.encoder_past_sum(x[i,:])
        #    if i == 0:
        #        sum_ptraj =  ptraj
        #    else:
        #        sum_ptraj = sum_ptraj + ptraj



        # ptraj = self.encoder_past(x)
        # print('ptraj:', ptraj.size())
        # print('dest:', dest.size())
        # print(ptraj.get_device())
        

        dest_enc = self.encoder_dest(dest)
        # print(dest_enc.get_device())
        # print('dest_enc:', dest_enc.size())
        # env_feat = torch.cat((ptraj.view(1,8), dest_enc), dim = 1)
        env_feat = torch.cat((ptraj, dest_enc), dim = 1)

        if not self.training:
            
            z = torch.Tensor(env_feat.size(0), self.zdim)
            z.normal_(0, self.sigma)
        else:
            # during training, use the destination to produce generated_dest and use it again to predict final future points

            # CVAE code
            ftraj = self.encoder_waypoint(waypoint)
            
            global_feat = torch.cat((env_feat, ftraj), dim = 1)
            latent =  self.encoder_latent(global_feat)

            mu = latent[:, 0:self.zdim] # 2-d array
            logvar = latent[:, self.zdim:] # 2-d array
            # print(logvar.size())

            var = logvar.mul(0.5).exp_()
            eps = torch.DoubleTensor(var.size()).normal_()
            eps = eps.to(device)
            z = eps.mul(var).add_(mu)
            # print(z.size())

        z = z.double().to(device)
        # print('here', env_feat.size())
        decoder_input = torch.cat((env_feat, z), dim = 1)
        # print(decoder_input.size())
        generated_waypoint = self.decoder(decoder_input)

        if self.training:
            # generated_waypoint_features = self.encoder_waypoint(generated_waypoint)

            # prediction_features = torch.cat((env_feat, generated_waypoint_features), dim = 1)
            

            # pred_future = self.predictor(prediction_features)
            return generated_waypoint , mu, logvar#, pred_future

        return generated_waypoint

