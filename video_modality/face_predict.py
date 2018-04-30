from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import keras
## VGGFACE MODEL
from keras.engine import  Model
from keras.layers import Concatenate, Add, Dropout, Flatten, Dense, Input, GlobalAveragePooling2D
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
from keras.callbacks import ModelCheckpoint

from keras.models import load_model
from keras.regularizers import l2


from keras.datasets import cifar10
import matplotlib.pyplot as plt
import cv2

import sys
sys.path.insert(0, '../data/')
from data import read_preprocessed_face_data

from myutils import my_ccc_loss, siamese_generator, myeval, prewhiten, get_aligned_indexes, preds2csv
import pandas as pd
import collections
import itertools

np.random.seed(1337)
__VIDEO_SEQ_LEN__ = 9
__N_CLASSES__     = 7
__EMOTIONS__      = True

__RES_FN__ = '/data/pmmf/OMG/VGGFACE/grid_test/'
__MODELS_FN__ = '/data/pmmf/OMG/VGGFACE/grid/'

if not os.path.exists(__RES_FN__):
    os.makedirs(__RES_FN__)

from keras import backend as K
print('image_data_format: ', K.image_data_format())


def myfit(model, x_train, y_train, x_val, y_val, lr=1e-04, loss_weights={'out_arousal': 1., 'out_valence': 1}, epochs=4, batch_size=64, weightspath="./tmp/weights_top.hdf5"):
    
    if __EMOTIONS__:
        # compile model 
        model.compile(optimizer=keras.optimizers.Adam(lr=lr), 
                                loss={'out_arousal': my_ccc_loss, 'out_valence': my_ccc_loss, 'out_categorical': keras.losses.categorical_crossentropy},
                                loss_weights=loss_weights)
    else:
        # compile model 
        model.compile(optimizer=keras.optimizers.Adam(lr=lr), 
                                loss={'out_arousal': my_ccc_loss, 'out_valence': my_ccc_loss},
                                loss_weights=loss_weights)

    # checkpoint
    checkpoint = ModelCheckpoint(weightspath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    # generators
    gen_train = siamese_generator(x_train, y_train, batch_size, augmentation=True, emotions=__EMOTIONS__)
    gen_val   = siamese_generator(x_val, y_val, batch_size, augmentation=False, emotions=__EMOTIONS__)

    x_train_len = x_train['frame_0'].shape[0]
    x_val_len   = x_val['frame_0'].shape[0]

    
    # fit
    model.summary()
    model.fit_generator(gen_train,
                        epochs=epochs,
                        validation_data=gen_val,
                        verbose=1,
                        steps_per_epoch=(x_train_len/batch_size),
                        validation_steps=(x_val_len/batch_size),
                        callbacks=callbacks_list)


    model.load_weights(weightspath)

    return model

def myVGGFACE(inputs, model='vgg16', n_dense=2, hidden_dim=512, l2_reg=1e-04, layer_name='pool5', train_mode=None):

    vgg_model = VGGFace(include_top=False, model=model, weights='vggface', input_shape=(96, 96, 3))
    out = vgg_model.get_layer(layer_name).output
    vgg_model_new = Model(vgg_model.input, out)
        
    for layer in vgg_model_new.layers:
        layer.trainable = False

    vgg_shared_streams = [vgg_model_new(i) for i in inputs]
    last_layer = Add()(vgg_shared_streams)
    x = GlobalAveragePooling2D()(last_layer)
    for i in range(n_dense):
        name='fc' + str(i + 6)
        x = Dense(hidden_dim, activation='relu', kernel_regularizer=l2(l2_reg), name=name)(x)
        x = Dropout(0.4)(x)
    out_arousal = Dense(1, activation='sigmoid', name='out_arousal')(x)
    out_valence = Dense(1, activation='tanh', name='out_valence')(x)
    if __EMOTIONS__:
        out_categorical = Dense(__N_CLASSES__, activation='softmax', name='out_categorical')(x)
        custom_vgg_model = Model(inputs, [out_arousal, out_valence, out_categorical])
    else:
        custom_vgg_model = Model(inputs, [out_arousal, out_valence])

    if train_mode is not None:
        for layer in vgg_model_new.layers[1:]:
            layer.trainable = True
            print(layer.name, layer.trainable)

    return custom_vgg_model

## READ DATA
train_fn = '../pre-process/Train_face_data.pckl'
val_fn   = '../pre-process/Validation_face_data.pckl'
test_fn  = '../pre-process/Test_face_data.pckl'

x_train, kpts_train, arousal_train, valence_train, emotion_train, groups_train, folder_train, \
x_val, kpts_val, arousal_val, valence_val, emotion_val, groups_val, folder_val, \
x_test, kpts_test, groups_test, folder_test = read_preprocessed_face_data(train_fn, val_fn, test_fn=test_fn)

x_train_len = x_train.shape[0] / __VIDEO_SEQ_LEN__
x_val_len   = x_val.shape[0] / __VIDEO_SEQ_LEN__
x_test_len  = x_test.shape[0] / __VIDEO_SEQ_LEN__

train_gt_fn = '/data/DB/OMG/omg_TrainVideos.csv'
val_gt_fn   = '/data/DB/OMG/omg_ValidationVideos.csv'
test_gt_fn  = '/data/DB/OMG/omg_TestVideos_WithoutLabels.csv'
aligned_val_indexes, gt_arousal, gt_valence = get_aligned_indexes(val_gt_fn, folder_val)
aligned_test_indexes                        = get_aligned_indexes(test_gt_fn, folder_test, test=True)

## PRE-PROCESSING
x_train = x_train.astype('float32') / 255
x_val   = x_val.astype('float32') / 255
x_test  = x_test.astype('float32') / 255

mean_train = np.mean(x_train, axis=(1,2),  keepdims=True)
mean_val   = np.mean(x_val, axis=(1,2),  keepdims=True)
mean_test  = np.mean(x_test, axis=(1,2),  keepdims=True)

x_train = x_train - mean_train
x_val   = x_val - mean_val
x_test  = x_test - mean_test

x_train_siamese = dict(('frame_'+str(i), x_train[i::__VIDEO_SEQ_LEN__,:,:,:]) for i in range(__VIDEO_SEQ_LEN__))
x_val_siamese   = dict(('frame_'+str(i), x_val[i::__VIDEO_SEQ_LEN__,:,:,:]) for i in range(__VIDEO_SEQ_LEN__))
x_test_siamese  = dict(('frame_'+str(i), x_test[i::__VIDEO_SEQ_LEN__,:,:,:]) for i in range(__VIDEO_SEQ_LEN__))



if __EMOTIONS__:
    emotion_train = keras.utils.to_categorical(emotion_train, __N_CLASSES__)
    emotion_val   = keras.utils.to_categorical(emotion_val, __N_CLASSES__)

    y_train = {'out_arousal': arousal_train, 'out_valence': valence_train, 'out_categorical': emotion_train}
    y_val   = {'out_arousal': arousal_val, 'out_valence': valence_val, 'out_categorical': emotion_val}
else:
    y_train = {'out_arousal': arousal_train, 'out_valence': valence_train}
    y_val   = {'out_arousal': arousal_val, 'out_valence': valence_val}

print(max(x_train.ravel()), min(x_train.ravel()), x_train.shape, x_val.shape, arousal_train.shape, valence_train.shape, arousal_val.shape, valence_val.shape)

## Grid parameters
grid_param = collections.OrderedDict()
grid_param = {'loss_weights': [{'out_arousal': 1., 'out_valence': 1., 'out_categorical': .1}, \
                               {'out_arousal': 1., 'out_valence': .5, 'out_categorical': .1}, \
                               {'out_arousal': 1., 'out_valence': .25, 'out_categorical': .05}], \
              'l2_reg': np.float32(np.logspace(-3, -5, 3)), \
              'n_dense': [2, 3], \
              'hidden_dim': [512, 1024]}

combinations = list(itertools.product(*(grid_param[k] for k in sorted(grid_param))))
print(">> GRID combos: ", len(combinations))
f_grid_fn =  __RES_FN__ + 'grid.log'
for idx, combo in enumerate(combinations):
    print('>> COMBO: ' + str(idx))
    dict_combo = {k: combo[i] for i, k in enumerate(sorted(grid_param))}
    print("combo " +  str(idx) + ': ', dict_combo)

    ## VGGFACE MODEL
    ## Predict top
    top_modelpath=__MODELS_FN__+ str(idx) + '_model_top.h5'
    vgg_top = load_model(top_modelpath, custom_objects={'my_ccc_loss': my_ccc_loss})

    # predictions
    preds = vgg_top.predict(x_test_siamese)
    preds_arousal = preds[0].ravel()[aligned_test_indexes]
    preds_valence = preds[1].ravel()[aligned_test_indexes]

    # save predtions to csv
    csv_fn = __RES_FN__ + str(idx) + '_preds_model_top.csv'
    preds2csv(csv_fn, test_gt_fn, preds_arousal, preds_valence)

    
    ## Predict full
    full_modelpath=__MODELS_FN__ + str(idx) + '_model_full.h5'
    vgg_full = load_model(full_modelpath, custom_objects={'my_ccc_loss': my_ccc_loss})

    # predictions
    preds = vgg_full.predict(x_test_siamese)
    preds_arousal = preds[0].ravel()[aligned_test_indexes]
    preds_valence = preds[1].ravel()[aligned_test_indexes]
        
    # save predtions to csv
    csv_fn = __RES_FN__ + str(idx) + '_preds_full_model.csv'
    preds2csv(csv_fn, test_gt_fn, preds_arousal, preds_valence)
    
    K.clear_session()
