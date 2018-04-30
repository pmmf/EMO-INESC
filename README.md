# EMO-INESC team 
This repository contains the source code of the EMO-INESC team solution for the OMG Emotion Recognition Challenge.

The implemented methodology is an ensemble of several models from two distinct modalities, namely video and text.


### Example Usage
#### Video Modality

1. Pre-processing:

- Convert videos to image frames.
~~~bash

python ./preprocessing/video2frames.py

~~~

- Read the annotations file and create an annotation file for each frame.
~~~bash

python ./preprocessing/get_annotations.py

~~~

- Face and facial landmarks detection.
~~~bash

python ./preprocessing/face_detection.py

~~~

- Selection of the video sequence length and number of frames per video to be used to design the models..
~~~bash

python ./preprocessing/face_channel.py

~~~


2. Face model fitting and prediction.
~~~bash

python ./video_modality/face_fit.py
python ./video_modality/face_predict.py

~~~


3. Facial landmarks model fitting and prediction.
~~~bash

python ./video_modality/landmarks_fit.py
python ./video_modality/landmarks_predict.py

~~~


#### Text Modality

- Fit and predict the traditional text model.
~~~bash

jupyter notebook ./text_modality/text_modality.ipynb

~~~

- Fit and predict the sequential text model.
~~~bash

python ./text_modality/text_modality.py

~~~

#### Ensemble
~~~bash

python ./ensemble/models_ensemble.py

~~~

### Library Versions
#### Video Modality
- Tensorflow v1.7.0
- Keras v2.1.5
- PyTorch v0.3.1.post2
- OpenCV v3.4.0
- Dlib v19.10.0
- [keras_vggface](https://github.com/rcmalli/keras-vggface)
- [face-alignment](https://github.com/1adrianb/face-alignment)

#### Text Modality
- Keras v2.1.1
- Tensorflow v1.4
- scikit-learn v0.19.1
- SciPy v1.0.0
- Natural Language Toolkit — NLTK v3.2.5
- TextBlob v0.15.1

### References
