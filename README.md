# EMO-INESC team 
This repository contains the source code of the EMO-INESC team solution for the OMG Emotion Recognition Challenge.

The implemented methodology is an ensemble of several models from two distinct modalities, namely video and text.


### Example Usage
#### Video Modality

1. Pre-processing:

- Conversion of the videos into image frames.
~~~bash

python video2frames.py

~~~

- Face and facial landmarks detection.
~~~bash

python pre-process.py

~~~

- Selection of the video sequence length and the number of frames per video.
~~~bash

python face_channel.py

~~~


2. Face model fitting and prediction.
~~~bash

python face_fit.py
python face_predict.py

~~~


3. Facial landmarks model fitting and prediction.
~~~bash

python landmarks_fit.py
python landmarks_predict.py

~~~


#### Text Modality




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

### References
