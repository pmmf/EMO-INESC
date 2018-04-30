
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
from skimage import io
import pickle
import matplotlib.pyplot as plt 
import numpy as np
import cv2

__RZE__         = 96.0
__OPT_FR__      = 30
__OPT_SEQ_LEN__ = 0.3
__OPT_NFRAMES__ = __OPT_SEQ_LEN__ * __OPT_FR__

__EXTENSION__ = '.pckl'
set_dir      = "/data/DB/OMG/Test_frames/"

video_list   = os.listdir(set_dir)
opt_nframes = []
abs_nframes = []
arousal_list = []
valence_list = []
emotion_list = []

crops_list = []
kpts_list  = []

groups_list = []

utterance_ctn = 0
folder_list = []
for idx, folder in enumerate(video_list): # loop over video folder
    utterances_list = sorted(os.listdir(os.path.join(*(set_dir, folder))))
    utterances_list = sorted(utterances_list,key=lambda x: int(x.split("_")[-1]))

    for u in utterances_list: # loop over utterance folder
        utterance_fn = os.path.join(*(set_dir,folder, u))
        
        # pckl files (crops + key-points)
        pckl_list = [f for f in os.listdir(utterance_fn) if f.endswith('.pckl')]
        pckl_list = sorted(pckl_list,key=lambda x: int(os.path.splitext(x)[-2].split("_")[-1]))

        # frames list (original images)
        frames_list = [f for f in os.listdir(utterance_fn) if f.endswith('.png')]
        frames_list = sorted(frames_list,key=lambda x: int(os.path.splitext(x)[-2].split("_")[-1]))

        # annotations
        with open(os.path.join(*(utterance_fn, 'annotations.gt')), 'rb') as f:
            annotations = pickle.load(f)
        
        if  len(pckl_list) == 0:
            print("NO FACES DETECTED", len(pckl_list), len(frames_list))
            input()
        else:
            utterance_ctn += 1
            groups_list.append(idx)
            folder_list.append(folder + ', ' + u)
           
            video_duration    = annotations['end'] - annotations['start']
            n_frames          = len(frames_list)
            frames_per_second = n_frames/video_duration

            abs_nframes.append(frames_per_second)

            opt_nframes = int(frames_per_second * __OPT_SEQ_LEN__)
            opt_nframes_central = int(opt_nframes/2.)
            n_frames_central = int(n_frames/2.)
            n_frames_central_copy = int(n_frames/2.)
            slice_1 = max(0, n_frames_central - opt_nframes_central)
            slice_2 = min(n_frames - 1, n_frames_central + opt_nframes_central)    
            print("BEFORE: ", n_frames_central,opt_nframes_central)  
            reset = True  
           
            while True:
                center_fn = os.path.join(*(utterance_fn, 'frame_' + str(n_frames_central) + '.pckl'))
                slice1_fn = os.path.join(*(utterance_fn, 'frame_' + str(int(slice_1)) + '.pckl'))

                if os.path.isfile(center_fn) and os.path.isfile(slice1_fn):
                    with open(slice1_fn, 'rb') as f:
                        my_crop1, _ = pickle.load(f)

                    with open(center_fn, 'rb') as f:
                        my_crop2, _ = pickle.load(f)

                    if len(my_crop1) != 0 and len(my_crop2) != 0:
                        break
                    else:
                        if n_frames_central == slice_2:                        
                            if reset:
                                n_frames_central = int( n_frames_central_copy - ((n_frames_central_copy) / 2.) )
                                reset = False
                            else:
                                n_frames_central = int( n_frames_central - ((n_frames_central) / 2.) )

                            slice_1 = max(0, n_frames_central - opt_nframes_central)
                            slice_2 = min(n_frames - 1, n_frames_central + opt_nframes_central) 

                        elif reset == False:
                            if n_frames_central == slice_1:
                                print('n_frames_central == slice_1')
                                input()
                                break
                            else:
                                n_frames_central = int( n_frames_central - ((n_frames_central) / 2.) )
                                slice_1 = max(0, n_frames_central - opt_nframes_central)
                                slice_2 = min(n_frames - 1, n_frames_central + opt_nframes_central)    

                        else:
                            n_frames_central = int( ((n_frames-n_frames_central) / 2.) + n_frames_central)
                            slice_1 = max(0, n_frames_central - opt_nframes_central)
                            slice_2 = min(n_frames - 1, n_frames_central + opt_nframes_central)        
                else:
                    if n_frames_central == slice_2:                        
                        if reset:
                            n_frames_central = int( n_frames_central_copy - ((n_frames_central_copy) / 2.) )
                            reset = False
                        else:
                            n_frames_central = int( n_frames_central - ((n_frames_central) / 2.) )

                        slice_1 = max(0, n_frames_central - opt_nframes_central)
                        slice_2 = min(n_frames - 1, n_frames_central + opt_nframes_central) 

                    elif reset == False:
                        if n_frames_central == slice_1:
                            print('n_frames_central == slice_1')
                            input()
                        else:
                            n_frames_central = int( n_frames_central - ((n_frames_central) / 2.) )
                            slice_1 = max(0, n_frames_central - opt_nframes_central)
                            slice_2 = min(n_frames - 1, n_frames_central + opt_nframes_central)    

                    else:
                        n_frames_central = int( ((n_frames-n_frames_central) / 2.) + n_frames_central)
                        slice_1 = max(0, n_frames_central - opt_nframes_central)
                        slice_2 = min(n_frames - 1, n_frames_central + opt_nframes_central)    
            
            fr_ratio   = frames_per_second / __OPT_FR__
            frame_list = [int(round(slice_1 + s * fr_ratio)) for s in range(int(__OPT_NFRAMES__))]     

            if len(frame_list) < __OPT_NFRAMES__:
                print("Different seq len")
                # input()

            crops_tmp = []
            kpts_tmp = []
            valid_frame = frame_list[0]
            print(frame_list)
            for fr in frame_list:
                pckl_fn = os.path.join(*(utterance_fn, 'frame_' + str(fr) + '.pckl'))

                if os.path.isfile(pckl_fn):
                    
                    with open(pckl_fn, 'rb') as f:
                        [my_crop, kpts_crop] = pickle.load(f)
                    
                    if len(my_crop) == 0:
                        print('EMPTY PCKL FILE', pckl_fn)
                        pckl_fn = os.path.join(*(utterance_fn, 'frame_' + str(valid_frame) + '.pckl'))
                        # input()

                    else:
                        pckl_fn = os.path.join(*(utterance_fn, 'frame_' + str(fr) + '.pckl'))
                        valid_frame = fr

                else:
                    pckl_fn = os.path.join(*(utterance_fn, 'frame_' + str(valid_frame) + '.pckl'))
                    print('INVALID MIDDLE FRAMES')
                    # input()


                with open(pckl_fn, 'rb') as f:
                    [my_crop, kpts_crop] = pickle.load(f)

                if my_crop.shape[1] == 0 or my_crop.shape[0] == 0:
                    image = io.imread(os.path.join(*(utterance_fn, 'frame_' + str(fr) + '.png')))

                    low_x = int(min(kpts_crop[:,0]))
                    sup_x = int(max(kpts_crop[:,0]))
                    low_y = int(min(kpts_crop[:,1]))
                    sup_y = int(max(kpts_crop[:,1]))

                    my_crop = image[low_y:sup_y, low_x:sup_x, :]
                    print('WARNING !!!! ')
                    input()

                # Resize
                factor_x, factor_y = __RZE__/ my_crop.shape[0], __RZE__/my_crop.shape[1]
                my_crop  = cv2.resize(my_crop, (int(__RZE__), int(__RZE__)), interpolation = cv2.INTER_CUBIC)
                kpts_crop[:,0] = kpts_crop[:,0] * factor_y 
                kpts_crop[:,1] = kpts_crop[:,1] * factor_x 
            
                crops_tmp.append(my_crop)
                kpts_tmp.append(kpts_crop)
            print('frame_list: ', frame_list, '; f/s: ', frames_per_second, '; opt #frames: ', opt_nframes)
            # for crop_idx, crop in enumerate(crops_tmp):
            #     plt.figure(1)
            #     plt.clf()
            #     plt.title(os.path.join(*(folder, u)))
            #     plt.imshow(crop)
            #     plt.plot(kpts_tmp[crop_idx][:,0], kpts_tmp[crop_idx][:,1], 'rx')
            #     plt.axis('off')
            #     plt.pause(0.1)
            #     plt.draw

            crops_list.append(crops_tmp)
            kpts_list.append(kpts_tmp)

crops_list   = np.array(crops_list)
kpts_list    = np.array(kpts_list)
# arousal_list = np.array(arousal_list)
# valence_list = np.array(valence_list)
# emotion_list = np.array(emotion_list)
groups_list  = np.array(groups_list)

print(crops_list[0].shape)
print(crops_list[1].shape)
print(folder_list[0])
# print(crops_list.shape, kpts_list.shape, arousal_list.shape, valence_list.shape, emotion_list.shape, groups_list.shape, len(folder_list))
print(crops_list.shape, kpts_list.shape, groups_list.shape, len(folder_list))

print(utterance_ctn)

# with open('Validation_face_data.pckl', 'wb') as f:
    # pickle.dump([crops_list, kpts_list, arousal_list, valence_list, emotion_list, groups_list, folder_list], f, protocol = 2)

with open('Test_face_data.pckl', 'wb') as f:
    pickle.dump([crops_list, kpts_list, groups_list, folder_list], f, protocol = 2)           
