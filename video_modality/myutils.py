from scipy.stats import pearsonr
import numpy as np
import pandas
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd
import os

def merge_preds(csv_list, weight_list, csv_fn, gt_file_fn):
    Npreds = len(pd.read_csv(csv_list[0])['video'])
    preds_arousal = np.zeros(Npreds)
    preds_valence = np.zeros(Npreds)
    arousal_weight_sum = np.zeros(Npreds)
    valence_weight_sum = np.zeros(Npreds)

    for csv_file, weight_tuple in zip(csv_list, weight_list):
        arousal_weight, valence_weight = weight_tuple

        df_sample = pd.read_csv(csv_file)
        mask = (df_sample['arousal']==df_sample['arousal'])
        preds_arousal[mask] += arousal_weight * df_sample['arousal'][mask]
        preds_valence[mask] += valence_weight * df_sample['valence'][mask]
        arousal_weight_sum[mask] += arousal_weight
        valence_weight_sum[mask] += valence_weight

    preds_arousal /= arousal_weight_sum
    preds_valence /= valence_weight_sum
  
    preds2csv(csv_fn, gt_file_fn, preds_arousal, preds_valence)

def preds2csv(csv_fn, gt_file_fn, preds_arousal, preds_valence):
    df_sample = pd.read_csv(gt_file_fn)
    columns = ['video', 'utterance', 'arousal', 'valence']
    data = {}
    data['video'] = df_sample['video']
    data['utterance'] = df_sample['utterance']
    data['arousal'] = preds_arousal
    data['valence'] = preds_valence
    df_preds = pd.DataFrame(data, columns=columns)
    df_preds.to_csv(csv_fn, sep=',', index=False)

def get_aligned_indexes(train_gt_fn, folder_train, test=False):
    df_train = pd.read_csv(train_gt_fn)
    # df_val   = pd.read_csv(val_gt_fn)

    gt_list = [df_train['video'][i] + ', ' + os.path.splitext(df_train['utterance'][i])[0] for i in range(len(df_train['video']))]
    aligned_indexes = [folder_train.index(value) for idx, value in enumerate(gt_list)]

    if test:
        return aligned_indexes
    else:
        return aligned_indexes, df_train['arousal'], df_train['valence']

# pre-whiten
def prewhiten(x, size):
    axis = (0, 1, 2)
    mean = np.mean(x, axis=axis, keepdims=True)
    std  = np.std(x, axis=axis, keepdims=True)
    std_adj = np.maximum(std, 1.0/np.sqrt(size))

    y = (x - mean) / std_adj
    return y

def mse(y_true, y_pred):
    from sklearn.metrics import mean_squared_error
    return mean_squared_error(y_true,y_pred)

def f1(y_true, y_pred):
    from sklearn.metrics import f1_score
    label = [0,1,2,3,4,5,6]
    return f1_score(y_true,y_pred,labels=label,average="micro")

def ccc(y_true, y_pred):
    true_mean = np.mean(y_true)
    true_variance = np.var(y_true)
    pred_mean = np.mean(y_pred)
    pred_variance = np.var(y_pred)

    rho,_ = pearsonr(y_pred,y_true)

    std_predictions = np.std(y_pred)

    std_gt = np.std(y_true)


    ccc = 2 * rho * std_gt * std_predictions / (
        std_predictions ** 2 + std_gt ** 2 +
        (pred_mean - true_mean) ** 2)

    return ccc, rho

def myeval(dataYArousal, dataYPredArousal, dataYValence, dataYPredValence):
    arousalCCC, acor = ccc(dataYArousal, dataYPredArousal)
    arousalmse = mse(dataYArousal, dataYPredArousal)
    valenceCCC, vcor = ccc(dataYValence, dataYPredValence)
    valencemse = mse(dataYValence, dataYPredValence)

    return arousalCCC,arousalmse, valenceCCC, valencemse

def my_ccc(y_true, y_pred):
    true_mean = K.mean(y_true)
    true_variance = K.var(y_true)
    pred_mean = K.mean(y_pred)
    pred_variance = K.var(y_pred)
    true_std = K.std(y_true)
    pred_std = K.std(y_pred)
    
    rho = K.sum((y_true - true_mean) * (y_pred - pred_mean)) / \
        (K.sqrt(K.sum(K.square(y_true - true_mean))) * 
         K.sqrt(K.sum(K.square(y_pred - pred_mean))) + 1e-15)
    
    std_predictions = K.std(y_pred)

    std_gt = K.std(y_true)


    ccc = (2 * rho * std_gt * std_predictions) / (
        K.square(std_predictions) + K.square(std_gt) +
        K.square(pred_mean - true_mean) + 1e-15)

    return ccc

def my_ccc_loss(y_true, y_pred):
    return - my_ccc(y_true, y_pred)


def cnn3d_generator(X,Y,minibatch_size,video_len=9,augmentation=True):
    
    keras_datagen = ImageDataGenerator(rotation_range=20,
                                       zoom_range=0.1,
                                       width_shift_range=0.25,
                                       height_shift_range=0.25,
                                       horizontal_flip=True,
                                       )
    N = X.shape[0]
    y_arousal = Y['out_arousal']
    y_valence = Y['out_valence']
    # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>SHAPE: ', N)
    while True:
        indices = np.random.choice(N, N, False)

        # faz os batches
        for i in range(N // minibatch_size):
            # X_list = {}

            # batch indices
            batch_indices = indices[i*minibatch_size:(i+1)*minibatch_size]

            # Y batches
            batch_arousal = y_arousal[batch_indices]
            batch_valence = y_valence[batch_indices]

            # Data batches
            batch_X = X[batch_indices]
            if augmentation:
                for fr in range(0, video_len):
                    batch_X_fr = batch_X[:, fr]
                    
                    batch_X_fr, _ = next(keras_datagen.flow(batch_X_fr,batch_valence, batch_size=minibatch_size, shuffle=False, seed = 1))
                    batch_X[:, fr] = batch_X_fr
                    

            yield(batch_X, {'out_arousal': batch_arousal, 'out_valence': batch_valence})

def siamese_kpts_generator(X,Y,minibatch_size,video_len=9,augmentation=True, emotions=False):
    
    # keras_datagen = ImageDataGenerator(rotation_range=20,
    #                                    zoom_range=0.1,
    #                                    width_shift_range=0.25,
    #                                    height_shift_range=0.25,
    #                                    horizontal_flip=True,
    #                                    )
    N = len(X['frame_0'])
    y_arousal = Y['out_arousal']
    y_valence = Y['out_valence']
    if emotions:
        y_categorical = Y['out_categorical']

    while True:
        indices = np.random.choice(N, N, False)

        # faz os batches
        for i in range(N // minibatch_size):
            X_list = {}

            # batch indices
            batch_indices = indices[i*minibatch_size:(i+1)*minibatch_size]

            # Y batches
            batch_arousal = y_arousal[batch_indices]
            batch_valence = y_valence[batch_indices]
            if emotions:
                batch_categorical = y_categorical[batch_indices]
   
            # Data batches
            for fr in range(0, video_len):
                frame_str = 'frame_' + str(fr)
                batch_X = X[frame_str][batch_indices]
                
                # if augmentation:
                #     batch_X, _ = next(keras_datagen.flow(batch_X,batch_valence, batch_size=minibatch_size, shuffle=False, seed = 1))
                X_list[frame_str] = batch_X
            # for im_idx in range(minibatch_size):
            #     fig_cnt = 0
            #     for ii in range(video_len):
            #         frame_str = 'frame_' + str(ii)
            #         plt.subplot(2,video_len,fig_cnt+1)
            #         plt.imshow(X_list[frame_str][im_idx])
            #         # plt.title(y_batch_original[im_idx*video_len+ii])
            #         plt.axis('off')
            #         plt.subplot(2,video_len,(fig_cnt+1)+video_len)
            #         plt.imshow(X_aug[frame_str][im_idx])
            #         # plt.title(y_batch[im_idx*video_len+ii])
            #         plt.axis('off')
            #         fig_cnt += 1
            # plt.show()
            if emotions:
                y_list = {'out_arousal': batch_arousal, 'out_valence': batch_valence, 'out_categorical': batch_categorical}
            else:
                y_list = {'out_arousal': batch_arousal, 'out_valence': batch_valence}
            yield(X_list, y_list)



def siamese_generator(X,Y,minibatch_size,video_len=9,augmentation=True, emotions=False):
    
    keras_datagen = ImageDataGenerator(rotation_range=20,
                                       zoom_range=0.1,
                                       width_shift_range=0.25,
                                       height_shift_range=0.25,
                                       horizontal_flip=True,
                                       )
    N = len(X['frame_0'])
    y_arousal = Y['out_arousal']
    y_valence = Y['out_valence']
    
    if emotions:
        y_categorical = Y['out_categorical']

    while True:
        indices = np.random.choice(N, N, False)

        # faz os batches
        for i in range(N // minibatch_size):
            X_list = {}

            # batch indices
            batch_indices = indices[i*minibatch_size:(i+1)*minibatch_size]

            # Y batches
            batch_arousal = y_arousal[batch_indices]
            batch_valence = y_valence[batch_indices]

            if emotions:
                batch_categorical = y_categorical[batch_indices]
   
            # Data batches
            for fr in range(0, video_len):
                frame_str = 'frame_' + str(fr)
                batch_X = X[frame_str][batch_indices]
                
                if augmentation:
                    batch_X, _ = next(keras_datagen.flow(batch_X,batch_valence, batch_size=minibatch_size, shuffle=False, seed = 1))
                X_list[frame_str] = batch_X
            # for im_idx in range(minibatch_size):
            #     fig_cnt = 0
            #     for ii in range(video_len):
            #         frame_str = 'frame_' + str(ii)
            #         plt.subplot(2,video_len,fig_cnt+1)
            #         plt.imshow(X_list[frame_str][im_idx])
            #         # plt.title(y_batch_original[im_idx*video_len+ii])
            #         plt.axis('off')
            #         plt.subplot(2,video_len,(fig_cnt+1)+video_len)
            #         plt.imshow(X_aug[frame_str][im_idx])
            #         # plt.title(y_batch[im_idx*video_len+ii])
            #         plt.axis('off')
            #         fig_cnt += 1
            # plt.show()
            if emotions:
                y_list = {'out_arousal': batch_arousal, 'out_valence': batch_valence, 'out_categorical': batch_categorical}
            else:
                y_list = {'out_arousal': batch_arousal, 'out_valence': batch_valence}
            yield(X_list, y_list)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '../data/')
    from data import read_preprocessed_face_data
    import pandas as pd

    __VIDEO_SEQ_LEN__ = 9

    ## READ DATA
    train_fn = '../pre-process/Train_face_data.pckl'
    val_fn   = '../pre-process/Validation_face_data.pckl'

    x_train, kpts_train, arousal_train, valence_train, emotion_train, groups_train, folder_train, \
    x_val, kpts_val, arousal_val, valence_val, emotion_val, groups_val, folder_val, = read_preprocessed_face_data(train_fn, val_fn)

    x_train_len = x_train.shape[0] / __VIDEO_SEQ_LEN__
    x_val_len   = x_val.shape[0] / __VIDEO_SEQ_LEN__


    x_train_siamese = dict(('frame_'+str(i), x_train[i::__VIDEO_SEQ_LEN__,:,:,:]) for i in range(__VIDEO_SEQ_LEN__))
    x_val_siamese   = dict(('frame_'+str(i), x_val[i::__VIDEO_SEQ_LEN__,:,:,:]) for i in range(__VIDEO_SEQ_LEN__))

    # print(folder_train)

    ## Align labels
    train_gt_fn = '/data/DB/OMG/omg_TrainVideos.csv'
    val_gt_fn   = '/data/DB/OMG/omg_ValidationVideos.csv'


    df_train = pd.read_csv(train_gt_fn)
    df_val   = pd.read_csv(val_gt_fn)

    aligned_indexes = get_aligned_indexes(val_gt_fn, folder_val)

    print(aligned_indexes)


    # for i in range(0, 100, 13):
    #     print(folder_train[i], emotion_train[i], valence_train[i], arousal_train[i])
    #     plt.imshow(x_train_siamese['frame_0'][i])
    #     plt.show()

    # for i in range(0, 100, 13):
    #     print(folder_val[i], emotion_val[i], valence_val[i], arousal_val[i])
    #     plt.imshow(x_val_siamese['frame_0'][i])
    #     plt.show()

    pass