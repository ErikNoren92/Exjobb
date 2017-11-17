import tensorflow as tf
import numpy as np
import argparse
import socket
import importlib
import time
import os
import scipy.misc
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider
import pc_util
import pdb
#importing the libraries

TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/kittiTest.txt'))
MODEL = importlib.import_module('pointnet_cls')
#Importing the test and the model

PC_SIZE = 128
BATCH_SIZE = 1
NUM_CLASSES = 3



def evaluate(is_training):
    #This is for the nodes
    with tf.device('/gpu:0'):
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, PC_SIZE)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        #Creating the label's field


        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        loss = MODEL.get_loss(pred, labels_pl, end_points)
        saver = tf.train.Saver()
        #calculate the loss, fetching the endpoints and save the system?.

        #HÄR OVAN ÄR ANROPET!!!!!!

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)
        #Preparing the session

        error_cnt = 0
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        total_seen_class = [0 for _ in range(NUM_CLASSES)]
        total_correct_class = [0 for _ in range(NUM_CLASSES)]
        #fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
        #preparing the estimation

        for PC in range(len(TEST_FILES)):
            current_data, current_label = provider.loadDataFile(TEST_FILES[PC])
            current_data = current_data[:, 0:PC_SIZE, :]
            current_label = np.squeeze(current_label)
            print(current_label)



if __name__ == '__main__':
    with tf.Graph().as_default():
        is_training = False
        evaluate(is_training)



