
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
import face_alignment
from skimage import io
import os

search_dir = "/data/DB/OMG/Test_frames/"
list_dir = os.listdir(search_dir)
print(len(list_dir))

files = [os.path.join(search_dir, f) for f in list_dir] # add path to each file
files.sort(key=lambda x: os.path.getmtime(x))

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, enable_cuda=True, flip_input=False)

print(len(files[0::2]))
print(files[0::2])

for idx, folder in enumerate(files):

    print(str(idx) + '/' + str(len(files)) + ': ' + folder )

    utterances_list = sorted(os.listdir(folder))

    for u in utterances_list:

        utterance_fn = os.path.join(*(folder, u))
        print(utterance_fn)
        preds = fa.process_folder(utterance_fn)
