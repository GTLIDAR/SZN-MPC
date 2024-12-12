import numpy as np
import numpy as np
# import visualization as vis
import pandas as pd
import numpy as np
import torch
# import random
# import os
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time
from read_txt_dataset import *


# With the test parameter, 0 for creating pkl for training and 1 for testing
def write_to_pkl(textfile, robot_batch, req_data_hist, req_future, radius, train, dataset):
    start_time = time.time()
    # for txtpath in textfile:
    if (train):
        # textfile_path = "../datatext/eth/train/" + txtpath
        x = get_seq_data_mpc(textfile, robot_batch, req_data_hist, req_future, radius)
        file_str = dataset + '_' + str(robot_batch) + '_' + str(req_data_hist) + '_' + str(req_future) + '_' + str(radius) + "_train"
    else:
        # textfile_path = "../datatext/eth/val/" + txtpath
        x = get_seq_data_mpc(textfile, robot_batch, req_data_hist, req_future, radius)
        file_str = dataset + '_' + str(robot_batch) + '_' + str(req_data_hist) + '_' + str(req_future) + '_' + str(radius) + "_test"
    
    with open("../datatext/" + file_str + ".pkl", 'wb') as f:
        pickle.dump(x, f)
        f.close()
    time_taken = time.time() - start_time
    print(f'The time it took to get sequence data without use of pkl was {time_taken:.4f} seconds')#X1 = get seq data (txt, params)
    return x

def read_from_pkl(textfile, robot_batch, req_data_hist, req_future, radius, train, dataset):
    start_time = time.time()
    if (train):
        file_str = dataset + '_' + str(robot_batch) + '_' + str(req_data_hist) + '_' + str(req_future) + '_' + str(radius) + "_train"
    else:
        file_str = dataset + '_' + str(robot_batch) + '_' + str(req_data_hist) + '_' + str(req_future) + '_' + str(radius) + "_test"
    print(file_str)
    with open("/home/ashamsah3/human_prediction/do-mpc/social_nav/datatext/"+ file_str + ".pkl", 'rb') as f:
        data = pickle.load(f)
        f.close()
    time_taken = time.time() - start_time
    print(f'The time it took to read from the pkl file was {time_taken:.4f} seconds')
    return data

def main():
    # textfile = ["biwi_hotel_train.txt", "crowds_zara02_train.txt", "crowds_zara03_train.txt", "students001_train.txt", "students003_train.txt", "uni_examples_train.txt", "biwi_eth_train.txt"]
    # textfile = ["biwi_hotel_train.txt", "crowds_zara01_train.txt", "crowds_zara02_train.txt", "crowds_zara03_train.txt", "uni_examples_train.txt", "biwi_eth_train.txt"]
    # textfile = ["biwi_hotel_val.txt", "crowds_zara02_val.txt", "crowds_zara03_val.txt", "students001_val.txt", "students003_val.txt", "uni_examples_val.txt", "biwi_eth_val.txt"]
    # textfile = ["biwi_hotel_val.txt", "crowds_zara01_val.txt", "crowds_zara02_val.txt", "crowds_zara03_val.txt", "uni_examples_val.txt", "biwi_eth_val.txt"]
    # textfile = ["biwi_hotel_val.txt", "crowds_zara01_val.txt", "crowds_zara02_val.txt", "crowds_zara03_val.txt", "uni_examples_val.txt", "biwi_eth_val.txt"]
    # textfile = ["biwi_hotel_train.txt"]
    textfile = ["crowds_zara01.txt"]
    batch_size = 20
    req_data_hist = 8
    req_future = 8
    radius = 4
    train = 0
    dataset = "Zara1_mpc"
    k = 0
    x = write_to_pkl(textfile, batch_size, req_data_hist, req_future, radius, train, dataset)
    # train_past_list, train_goal_list, train_waypoint_list, future_list, train_past_list_g, train_goal_list_g, train_waypoint_list_g = read_from_pkl(dataset, batch_size, req_data_hist, req_future, radius, train, dataset)
    # for i, (train_past, train_goal, train_waypoint) in enumerate(zip(train_past_list, train_goal_list, train_waypoint_list)):
    #     print(train_past.size())
    #     print(i)
        # k = k + train_past.size(0)
    # print(len(train_past_list))
    # print(k)
    # print(k/7)


# main()

#X2 = function2(Y)


#X1 == X2 