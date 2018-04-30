
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
from skimage import io
import pickle
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../models/')
from myutils import preds2csv, merge_preds, myeval, get_aligned_indexes


def get_weights_list(models_list, gt_file_fn):
    df_gt = pd.read_csv(gt_file_fn)
    weight_list = []
    final_models_list = []
    for idx, model_fn in enumerate(models_list):
        df_model = pd.read_csv(model_fn)
        mask = (df_model['arousal']==df_model['arousal'])


        arousalCCC,arousalmse, valenceCCC, valencemse = myeval(df_gt['arousal'][mask], df_model['arousal'][mask], df_gt['valence'][mask], df_model['valence'][mask])

        # if arousalCCC > 0.20 or valenceCCC > 0.30:
        weight_list.append((arousalCCC, valenceCCC))
        final_models_list.append(model_fn)

    
    return weight_list, final_models_list


if __name__ == "__main__":

    # GT files
    train_gt_fn = '/data/DB/OMG/omg_TrainVideos.csv'
    val_gt_fn   = '/data/DB/OMG/omg_ValidationVideos.csv'
    test_gt_fn  = '/data/DB/OMG/omg_TestVideos_WithoutLabels.csv'

    df_gt       = pd.read_csv(test_gt_fn)

    # models list to ensemble
    ensemble_folder_fn = './test_ensemble/'
    csv_list = sorted(os.listdir(ensemble_folder_fn)) 
    csv_list = [os.path.join(*(ensemble_folder_fn, model)) for model in csv_list]
    print(len(csv_list), csv_list)

    # weights list
    # weight_list, csv_list = get_weights_list(models_list=csv_list, gt_file_fn=val_gt_fn)

    # with open('weight_list.pckl', 'wb') as f:
    #     pickle.dump(weight_list, f, protocol = 2)

    with open('weight_list.pckl', 'rb') as f:
        weight_list = pickle.load(f)

    print(len(csv_list), csv_list)

    # ensemble predictions
    csv_fn = 'emo-inesc-predictions.csv'
    merge_preds(csv_list, weight_list, csv_fn, gt_file_fn=test_gt_fn)
    
    # check losses on validation
    # df_ensemble = pd.read_csv(csv_fn)

    # arousalCCC,arousalmse, valenceCCC, valencemse = myeval(df_gt['arousal'], df_ensemble['arousal'], df_gt['valence'], df_ensemble['valence'])
    # print("vgg_full: ", arousalCCC,arousalmse, valenceCCC, valencemse)







