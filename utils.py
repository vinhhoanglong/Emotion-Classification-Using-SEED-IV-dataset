import os
import numpy as np
from scipy.io import loadmat

def map_coor():
    channel_coords = [['0', '0', 'AF3', 'FP1', 'FPZ', 'FP2', 'AF4', '0', '0'], ['F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8'], ['FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8'], ['T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8'], ['TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8'], ['P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8'], ['0', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', '0'], ['0', '0', 'CB1', 'O1', 'OZ', 'O2', 'CB2', '0', '0']]
    channel_list = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ', 'O2', 'CB2']
    print(len(channel_coords), len(channel_coords[0]))
    coord_dict = {}
    for n in range(len(channel_list)):
        for i, l in enumerate(channel_coords):
            for j, x in enumerate(l):
                if (channel_list[n] == x):
                    coord_dict[n] = (i,j)
    return coord_dict

def load_data(directories: str , coord_dict):   # dir : data/eeg_feature_smooth/*
    n = 24
    perSample = ['de_movingAve', 'de_LDS', 'psd_movingAve', 'psd_LDS']
    array = np.zeros(shape=(len(directories),len(os.listdir(directories[0])), n, 4, 8, 9, 5, 64)) # features = 4 datatypes*(8 x 9 eeg channel locs)*5 frequency bands*64 timestamps(zero padded) // trials = (3 sessions) x 15 people x 24 labels
    li = []
    for h, dire in enumerate(directories):
        print(dire)
        data = [loadmat(dire + file) for file in os.listdir(dire)]
        for i, bigsample in enumerate(data):
            for j in range(n):
                for k, key in enumerate(perSample):
                    sample = np.transpose(np.array(bigsample[key + str(j+1)]), (0,2,1))
                    sample = np.pad(sample, [(0,0), (0,0), (0, 64-sample.shape[2])])
                    for l, channel in enumerate(sample):
                        array[h][i][j][k][coord_dict[l][0]][coord_dict[l][1]] = channel

    return array
