"""
    1CM290 Maintenance Optimziation and Engineering (Lecturer: J. Lee)
    Assignment: Data Cahllenges 2025
    Challenge: Detection of faults in gears.
    This is a template for the assignment.
    You may fill the parts <YOUR CODE HERE> or add new parts as you need.
    You may change as you need.
    Please add concise and comprehensive comments.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


# Read Raw Data
def read_raw_data(path):
    data_dict = {}  # Data of each sample is stored in dictionary [i, h, v, n]
    for file in os.listdir(path):
        print(file)  # Check the file name
        key = file.split('.')[0].split('_')
        i = int(key[0])  # i: index of data
        h = int(key[1])  # h: label of data (0: normal, 1: fault)
        v = int(key[2].split('V')[1])  # v: rotational speed
        n = int(key[3].split('N')[0])  # n: loads
        key = (i, h, v, n)
        data_dict[key] = pd.read_csv(path + file)
    return data_dict


data_train_raw = read_raw_data('data_gear/train/')
data_valid_raw = read_raw_data('data_gear/valid/')
data_test_raw = read_raw_data('data_gear/test/')

# Task (a) Preprocess the raw data
def preprocess(data_dict):
    sample_rate = 20480
    data_dict_fft = {}  # Data of each sample is stored in dictionary [i, h, v, n]
    for i, h, v, n in data_dict:
        # YOUR CODE HERE
        continue
    return data_dict_fft


data_train = preprocess(data_train_raw)
data_valid = preprocess(data_valid_raw)
data_test = preprocess(data_test_raw)


# Save preprocessed Data
def save_dataset_npy(data_dict_fft, file_name):
    label = []
    feature = []
    for i, h, v, n in data_dict_fft:
        label.append(np.array([h]))  # Label of data (0: normal, 1: fault)
        feature.append(np.hstack([data_dict_fft[i, h, v, n][1],
                                   data_dict_fft[i, h, v, n][2],
                                   data_dict_fft[i, h, v, n][3]]))
    np.save(file_name + '_label.npy', np.array(label))
    np.save(file_name + '_feature.npy', np.array(feature))


save_dataset_npy(data_train, 'data_gear_train')
save_dataset_npy(data_valid, 'data_gear_valid')
save_dataset_npy(data_test, 'data_gear_test')


# YOUR CODE HERE
