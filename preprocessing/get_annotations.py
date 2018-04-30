
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
from skimage import io
import pickle
import matplotlib.pyplot as plt 
import pandas as pd


set_dir = "/data/DB/OMG/Test_frames/"

# Read csv
file_fn = "/data/DB/OMG/omg_TestVideos_WithoutLabels.csv"
df = pd.read_csv(file_fn)

start     = df['start']
end       = df['end']
video     = df['video']
utterance = df['utterance']
# arousal   = df['arousal']
# valence   = df['valence']
# emotion   = df['EmotionMaxVote']


# save an annotation file for each frame
print("NUMBER OF VIDEOS :: ", len(start))
for idx in range(len(start)):
    print(idx)

    annotations = {'start': start[idx], \
                    'end': end[idx]}
    
    # annotations = {'start': start[idx], \
    #                'end': end[idx], \
    #                'arousal': arousal[idx], \
    #                'valence': valence[idx], \
    #                'emotion': emotion[idx]}
        
    # save to file
    file_fn = os.path.join(*(set_dir,video[idx],utterance[idx].split('.')[0], 'annotations.gt'))

    if os.path.isdir( os.path.join(*(set_dir,video[idx],utterance[idx].split('.')[0])) ):
        print('>> ', idx, video[idx], utterance[idx].split('.')[0])
        
        with open(file_fn, 'wb') as f:
            pickle.dump(annotations, f, protocol = 2)
    else:
        print(">> ERROR : CURRENT DIR DOES NOT EXIST!", idx, video[idx], utterance[idx].split('.')[0])
        input()






