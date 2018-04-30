from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# os.environ["CUDA_VISIBLE_DEVICES"]="2"

import numpy as np
import keras
## VGGFACE MODEL
from keras.engine import  Model
from keras.layers import Concatenate, Add, Dropout, Flatten, Dense, Input, GlobalAveragePooling2D, GaussianDropout
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

from myutils import my_ccc_loss, siamese_kpts_generator, myeval, prewhiten, get_aligned_indexes, preds2csv
import pandas as pd
import collections
import itertools

np.random.seed(1337)
__VIDEO_SEQ_LEN__ = 9
__N_CLASSES__     = 7
__EMOTIONS__      = True

__RES_FN__    = '/data/pmmf/OMG/KPTS/grid_wo_norm_mlayers_test/'
__MODELS_FN__ = '/data/pmmf/OMG/KPTS/grid_wo_norm_mlayers/'

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
    gen_train = siamese_kpts_generator(x_train, y_train, batch_size, augmentation=True, emotions=__EMOTIONS__)
    gen_val   = siamese_kpts_generator(x_val, y_val, batch_size, augmentation=False, emotions=__EMOTIONS__)

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

def siamese_encoder(input_shape, n_dense=2, hidden_dim=256, l2_reg=1e-04):
    base = base_encoder(input_shape, hidden_dim, l2_reg)
    inputs  = [Input(shape=input_shape, name='frame_'+str(i)) for i in range(__VIDEO_SEQ_LEN__)]

    shared_streams = [base(i) for i in inputs]
    x = Concatenate()(shared_streams)
    for i in range(n_dense):
        name='fc' + str(i + 3)
        x = Dense(hidden_dim//4, activation='relu', kernel_regularizer=l2(l2_reg), name=name)(x)
        x = Dropout(0.4)(x)
    out_arousal = Dense(1, activation='sigmoid', name='out_arousal')(x)
    out_valence = Dense(1, activation='tanh', name='out_valence')(x)
    if __EMOTIONS__:
        out_categorical = Dense(__N_CLASSES__, activation='softmax', name='out_categorical')(x)
        siamese_encoder = Model(inputs, [out_arousal, out_valence, out_categorical])
    else:
        siamese_encoder = Model(inputs, [out_arousal, out_valence])
    siamese_encoder.summary()
    return siamese_encoder

def base_encoder(input_shape, hidden_dim, l2_reg):
    input = Input(shape=input_shape)
    x = Dense(hidden_dim, activation='relu', input_shape=input_shape, kernel_regularizer=l2(l2_reg))(input)
    x = Dense(hidden_dim//2, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    return Model(input, x)

def mydiff(d):
    diff = {}
    for ii in range(1, __VIDEO_SEQ_LEN__):
        if ii == 1:
            diff['frame_0'] = d['frame_'+str(ii)] - d['frame_'+str(ii-1)]
        diff['frame_'+str(ii)] = d['frame_'+str(ii)] - d['frame_'+str(ii-1)]
    return diff

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

## COMPUTE KPTS-FEATS
from feat_kpts_geometry import feat_kpts_geometry
from sklearn.preprocessing import StandardScaler

feat_kpts   = feat_kpts_geometry()
# train
train_feats = feat_kpts.describe(kpts_train)
scaler = StandardScaler()
scaler.fit(train_feats)
train_feats = scaler.transform(train_feats)

# val
val_feats = feat_kpts.describe(kpts_val)
val_feats = scaler.transform(val_feats)

# test
test_feats = feat_kpts.describe(kpts_test)
test_feats = scaler.transform(test_feats)

## PRE-PROCESSING
x_train = x_train.astype('float32') / 255
x_val   = x_val.astype('float32') / 255
x_test  = x_test.astype('float32') / 255

kpts_train = kpts_train.astype('float32') / 96
kpts_val   = kpts_val.astype('float32') / 96
kpts_test  = kpts_test.astype('float32') / 96

kpts_train_siamese = dict(('frame_'+str(i), kpts_train[i::__VIDEO_SEQ_LEN__,:,:2].reshape(-1, 68*2)) for i in range(__VIDEO_SEQ_LEN__))
kpts_val_siamese   = dict(('frame_'+str(i), kpts_val[i::__VIDEO_SEQ_LEN__,:,:2].reshape(-1, 68*2)) for i in range(__VIDEO_SEQ_LEN__))
kpts_test_siamese   = dict(('frame_'+str(i), kpts_test[i::__VIDEO_SEQ_LEN__,:,:2].reshape(-1, 68*2)) for i in range(__VIDEO_SEQ_LEN__))

train_feats_siamese = dict(('frame_'+str(i), train_feats[i::__VIDEO_SEQ_LEN__,:]) for i in range(__VIDEO_SEQ_LEN__))
val_feats_siamese   = dict(('frame_'+str(i), val_feats[i::__VIDEO_SEQ_LEN__,:]) for i in range(__VIDEO_SEQ_LEN__))
test_feats_siamese  = dict(('frame_'+str(i), test_feats[i::__VIDEO_SEQ_LEN__,:]) for i in range(__VIDEO_SEQ_LEN__))


diff_train  = mydiff(kpts_train_siamese)
diff_val    = mydiff(kpts_val_siamese)
diff_test   = mydiff(kpts_test_siamese)

diff2_train = mydiff(mydiff(kpts_train_siamese))
diff2_val   = mydiff(mydiff(kpts_val_siamese))
diff2_test  = mydiff(mydiff(kpts_test_siamese))

combo_train = {}
combo_val   = {}
combo_test  = {}
for ii in range(__VIDEO_SEQ_LEN__):
    combo_train['frame_'+str(ii)] = np.concatenate((kpts_train_siamese['frame_'+str(ii)], diff_train['frame_'+str(ii)], diff2_train['frame_'+str(ii)], train_feats_siamese['frame_'+str(ii)]), axis=1)
    combo_val['frame_'+str(ii)]   = np.concatenate((kpts_val_siamese['frame_'+str(ii)], diff_val['frame_'+str(ii)], diff2_val['frame_'+str(ii)], val_feats_siamese['frame_'+str(ii)]), axis=1)
    combo_test['frame_'+str(ii)]  = np.concatenate((kpts_test_siamese['frame_'+str(ii)], diff_test['frame_'+str(ii)], diff2_test['frame_'+str(ii)], test_feats_siamese['frame_'+str(ii)]), axis=1)

if __EMOTIONS__:
    emotion_train = keras.utils.to_categorical(emotion_train, __N_CLASSES__)
    emotion_val   = keras.utils.to_categorical(emotion_val, __N_CLASSES__)

    y_train = {'out_arousal': arousal_train, 'out_valence': valence_train, 'out_categorical': emotion_train}
    y_val   = {'out_arousal': arousal_val, 'out_valence': valence_val, 'out_categorical': emotion_val}
else:
    y_train = {'out_arousal': arousal_train, 'out_valence': valence_train}
    y_val   = {'out_arousal': arousal_val, 'out_valence': valence_val}

## Grid parameters
grid_param = collections.OrderedDict()
grid_param = {'loss_weights': [{'out_arousal': 1., 'out_valence': 1., 'out_categorical': .1}, \
                               {'out_arousal': 1., 'out_valence': .5, 'out_categorical': .1}, \
                               {'out_arousal': .5, 'out_valence': .1, 'out_categorical': .1}, \
                               {'out_arousal': 1., 'out_valence': .25, 'out_categorical': .05}, \
                               {'out_arousal': .25, 'out_valence': .1, 'out_categorical': .05}], \
              'l2_reg': np.float32(np.logspace(-3, -5, 3)),\
              'n_dense': [1, 2, 3], \
              'hidden_dim': [256, 512]}

combinations = list(itertools.product(*(grid_param[k] for k in sorted(grid_param))))
print(">> GRID combos: ", len(combinations))
f_grid_fn =  __RES_FN__ + 'grid_kpts_wo_norm.log'
for idx, combo in enumerate(combinations):
    print('>> COMBO: ' + str(idx))
    dict_combo = {k: combo[i] for i, k in enumerate(sorted(grid_param))}
    print("combo " +  str(idx) + ': ', dict_combo)

    ## MLP MODEL
    top_modelpath= __MODELS_FN__ + str(idx) + '_model_top_ktps.h5'
    siamese_mlp = load_model(top_modelpath, custom_objects={'my_ccc_loss': my_ccc_loss})

    # predictions
    preds = siamese_mlp.predict(combo_test)
    preds_arousal = preds[0].ravel()[aligned_test_indexes]
    preds_valence = preds[1].ravel()[aligned_test_indexes]

    # save predtions to csv
    csv_fn = __RES_FN__ + str(idx) + '_preds_model_top_ktps.csv'
    preds2csv(csv_fn, test_gt_fn, preds_arousal, preds_valence)
    
    K.clear_session()
