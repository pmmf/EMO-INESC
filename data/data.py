
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
from skimage import io
import pickle
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np


def read_preprocessed_face_data(train_fn, val_fn, test_fn=False, data_3D=False):

    # read prerpocessed training data
    with open(train_fn, 'rb') as f:
        [x_train, \
         kpts_train, \
         arousal_train, \
         valence_train, \
         emotion_train, \
         groups_train, \
         folder_train] = pickle.load(f)

    print(kpts_train.shape)
    video_seq_len = x_train.shape[1]

    if not data_3D:
        x_train = x_train.reshape(-1, *x_train.shape[2:])
        kpts_train = kpts_train.reshape(-1, *kpts_train.shape[2:])
    print('Training set size: ', x_train.shape, kpts_train.shape, arousal_train.shape, valence_train.shape, emotion_train.shape, len(folder_train))
    
    # read preprocessed validation data
    with open(val_fn, 'rb') as f:
        [x_val, \
         kpts_val, \
         arousal_val, \
         valence_val, \
         emotion_val, \
         groups_val, \
         folder_val] = pickle.load(f)

    if not data_3D:
        x_val = x_val.reshape(-1, *x_val.shape[2:])
        kpts_val = kpts_val.reshape(-1, *kpts_val.shape[2:])
    print('Validation set size: ', x_val.shape, kpts_val.shape, arousal_val.shape, valence_val.shape, emotion_val.shape, len(folder_val))    
    

    if test_fn:
        # read preprocessed test data
        with open(test_fn, 'rb') as f:
            [x_test, \
            kpts_test, \
            groups_test, \
            folder_test] = pickle.load(f)

        if not data_3D:
            x_test = x_test.reshape(-1, *x_test.shape[2:])
            kpts_test = kpts_test.reshape(-1, *kpts_test.shape[2:])
        print('Test set size: ', x_test.shape, kpts_test.shape, len(folder_test))


        return x_train, kpts_train, arousal_train, valence_train, emotion_train, groups_train, folder_train, \
            x_val, kpts_val, arousal_val, valence_val, emotion_val, groups_val, folder_val,\
            x_test, kpts_test, groups_test, folder_test 

    return x_train, kpts_train, arousal_train, valence_train, emotion_train, groups_train, folder_train, \
            x_val, kpts_val, arousal_val, valence_val, emotion_val, groups_val, folder_val



if __name__ == "__main__":

    ## READ DATA
    train_fn = '../pre-process/Train_face_data.pckl'
    val_fn   = '../pre-process/Validation_face_data.pckl'

    x_train, kpts_train, arousal_train, valence_train, emotion_train, groups_train, folder_train, \
    x_val, kpts_val, arousal_val, valence_val, emotion_val, groups_val, folder_val, = read_preprocessed_face_data(train_fn, val_fn)

    # for idx, im in enumerate(x_train):
    #     print(folder_train[idx], arousal_train[idx], valence_train[idx])
    #     plt.figure(1)
    #     plt.clf()
    #     # plt.title(folder_train + '; arousal: ' + str(arousal_train[idx]) + '; valence: ' + str(valence_train[idx]))
    #     plt.imshow(im)
    #     plt.plot(kpts_train[idx][:,0], kpts_train[idx][:,1], 'rx')
    #     plt.axis('off')
    #     plt.pause(0.1)
    #     plt.draw

    for idx, im in enumerate(x_val):
        print(folder_val[idx], arousal_val[idx], valence_val[idx])
        plt.figure(1)
        plt.clf()
        # plt.title(folder_train + '; arousal: ' + str(arousal_train[idx]) + '; valence: ' + str(valence_train[idx]))
        plt.imshow(im)
        plt.plot(kpts_val[idx][:,0], kpts_val[idx][:,1], 'rx')
        plt.axis('off')
        plt.pause(0.1)
        plt.draw







