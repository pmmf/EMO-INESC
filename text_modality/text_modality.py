import pandas as pd
import numpy as np
import os
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense, Concatenate, Reshape, Lambda, Dropout
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import mean_squared_error, categorical_crossentropy
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.metrics import categorical_accuracy
from scipy.stats import pearsonr

TRAIN_DATA = "./text_train_data.csv"
VALID_DATA = "./text_valid_data.csv"
TEST_DATA = "./text_test_data.csv"
GLOVE_DIR = "/data/DB/glove.6B/"
MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 200
EMBEDDING_DIM = 50
# SEED = 123
# np.random.seed(SEED)

def get_data(path):
  # Read csv
  df = pd.read_csv(path)
  
  transcript = df['transcript']
  arousal    = df['arousal']
  valence    = df['valence']
  emotion    = df['EmotionMaxVote']
  
  # Discard videos where transcript is empty (nan)
  keep_idx = [i for i in range(len(transcript)) if transcript[i]==transcript[i]]
  transcript = [transcript[i] for i in keep_idx]
  arousal = [arousal[i] for i in keep_idx]
  valence = [valence[i] for i in keep_idx]
  emotion = [emotion[i] for i in keep_idx]
  
  arousal = np.array(arousal).reshape(-1,1)
  valence = np.array(valence).reshape(-1,1)
  emotion = np.array(emotion).astype(int)
  
  ycont = np.concatenate([arousal, valence], axis=1)
  ydisc = emotion
  
  return transcript, ycont, ydisc, keep_idx


def keras_ccc(y_true, y_pred):
  true_mean = K.mean(y_true)
  true_variance = K.var(y_true)
  pred_mean = K.mean(y_pred)
  pred_mean = K.stop_gradient(pred_mean)
  
  eps = 1e-15
  
  rho = K.sum((y_true - true_mean) * (y_pred - pred_mean)) / \
      (K.sqrt(K.sum(K.square(y_true - true_mean))) * 
       K.sqrt(K.sum(K.square(y_pred - pred_mean))) + eps)
  
  std_predictions = K.std(y_pred)
  
  std_gt = K.std(y_true)
  
  
  ccc = (2 * rho * std_gt * std_predictions) / (
      K.square(std_predictions) + K.square(std_gt) +
      K.square(pred_mean - true_mean) + eps)
  
  return ccc

def ccc_loss(y_true, y_pred):
  return -1*keras_ccc(y_true, y_pred)


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


# prepare training and validation data
text_train, ycont_train, ydisc_train, _ = get_data(TRAIN_DATA)
text_valid, ycont_valid, ydisc_valid, _ = get_data(VALID_DATA)
text_test, _, _, keep_idx_test = get_data(TEST_DATA)

ydisc_train = to_categorical(ydisc_train)
ydisc_valid = to_categorical(ydisc_valid)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(text_train +  text_valid + text_test)
sequences_train = tokenizer.texts_to_sequences(text_train)
sequences_valid = tokenizer.texts_to_sequences(text_valid)
sequences_test = tokenizer.texts_to_sequences(text_test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

Xtrain = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
Xvalid = pad_sequences(sequences_valid, maxlen=MAX_SEQUENCE_LENGTH)
Xtest = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

####

# load pre-trained embeddings
embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.50d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
  embedding_vector = embeddings_index.get(word)
  if embedding_vector is not None:
    # words not found in embedding index will be all-zeros.
    embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
###

# define the model and train it

# WITH CATEGORICAL OUTPUT
def create_model():
  w_reg = 1e-4
  
  sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
  embed = embedding_layer(sequence_input)
  h_lstm = LSTM(16, return_sequences=True, activity_regularizer=l2(w_reg))(embed)
  h_lstm = LSTM(16, activity_regularizer=l2(w_reg))(h_lstm)
  h_dense = Dense(2+7, activity_regularizer=l2(w_reg))(h_lstm)
  arousal_pred = Lambda(lambda x: K.sigmoid(x[:, 0]))(h_dense) 
  valence_pred = Lambda(lambda x: K.tanh(x[:, 1]))(h_dense)
  emotion_pred = Lambda(lambda x: K.softmax(x[:, 2::]), name='emotion_pred')(h_dense)
  arousal_pred = Reshape((1,), name='arousal_pred')(arousal_pred)
  valence_pred = Reshape((1,), name='valence_pred')(valence_pred)
  
  model = Model(inputs=sequence_input, outputs=[arousal_pred, valence_pred, emotion_pred])
  model.compile(optimizer=keras.optimizers.Adam(lr=5e-4), 
                loss=[ccc_loss, ccc_loss, categorical_crossentropy], 
                loss_weights=[1, 1., 0.5],
                )               
                
  print(model.summary())
  
  return model

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
checkpointer = ModelCheckpoint(filepath='./tmp/weights.hdf5', monitor='val_loss', verbose=1, save_best_only=True)

model = create_model()
model.fit(Xtrain, [ycont_train[:,0], ycont_train[:,1], ydisc_train], epochs=100, batch_size=128,
          validation_data=(Xvalid, [ycont_valid[:,0], ycont_valid[:,1], ydisc_valid]), callbacks=[earlystop, checkpointer], verbose=2)
model.load_weights('./tmp/weights.hdf5')

arousal_pred, valence_pred = model.predict(Xvalid)[0:2]
ccc_arousal = ccc(ycont_valid[:,0], arousal_pred.reshape(-1))[0]
ccc_valence = ccc(ycont_valid[:,1], valence_pred.reshape(-1))[0]
print("ccc_arousal   ",ccc_arousal)
print("ccc_valence   ",ccc_valence)

model.save('./tmp/model.hdf5')

arousal_pred, valence_pred = model.predict(Xtest)[0:2]
arousal_pred = arousal_pred.reshape(-1)
valence_pred = valence_pred.reshape(-1)
Ntest = 2229
arousal_pred_full = np.float('nan')*np.ones(Ntest)
valence_pred_full = np.float('nan')*np.ones(Ntest)
arousal_pred_full[keep_idx_test] = arousal_pred
valence_pred_full[keep_idx_test] = valence_pred
preds2csv('./test_preds.csv', './omg_TestVideos_WithoutLabels.csv', arousal_pred_full, valence_pred_full)
