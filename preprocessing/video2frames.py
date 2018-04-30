import os
import cv2
import matplotlib.pyplot as plt

if __name__ == "__main__":
    videos_fn   = '/data/DB/OMG/Test_videos/'
    frames_fn   = '/data/DB/OMG/Test_frames/'
    videos_list = sorted(os.listdir(videos_fn))[:-1]
    for video in videos_list:
        utterances_fn   = os.path.join(*(videos_fn, video))
        utterances_list = sorted(os.listdir(utterances_fn))
        print("utterances_list: ", videos_list)
        
        for utterance in utterances_list:
            utterance_fn = os.path.join(*(utterances_fn, utterance))
            print(utterance_fn)

            res_fn = os.path.join(*(frames_fn, video, os.path.splitext(utterance)[0]))
            if not os.path.exists(res_fn):
                os.makedirs(res_fn)

            vidcap = cv2.VideoCapture(utterance_fn)
            success,image = vidcap.read()
            count = 0
            success = True
            while success:

                idx = 'frame_' + str(count) + '.png'
                frame_fn = os.path.join(*(frames_fn, video, os.path.splitext(utterance)[0], idx))

                cv2.imwrite(frame_fn, image)   
                success,image = vidcap.read()
                count += 1

