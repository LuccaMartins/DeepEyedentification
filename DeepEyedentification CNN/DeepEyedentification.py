# -*- coding: utf-8 -*-
''' 
DeepEyedentification: Biometric Identification using Micro-movements of the Eye

Once this paper has been accepted, we will publish the full 
data set (75 readers, 12 texts) and all code of this project.

Here, we provide an example of how our trained model is applied in a 
multi-class setting. We provide the model architecture and the weights of a 
trained model on one specific hold-out split. We only provide the reading 
data of all persons on the held-out text, but not the training data. 
'''


import numpy as np
from keras.models import Model
from keras.models import model_from_json
from matplotlib import pyplot as plt

num_persons = 75 # number of classes for multi-class setting
num_secs = 10 # number of gaze-velocity seconds up to which predictions are computed

# load neural model
# input:
#    jsonPath: path to load model.json
#    h5Path: path to load model.h5
# output:
#    return keras model
def load_model(jsonPath,h5Path):
    # loading model
    model = model_from_json(open(jsonPath, encoding='utf-8').read())
    model.load_weights(h5Path)
    return model


# transformation for slow subnet
# input:
    # x: gaze velocity data in °/ms
    # f: scaling factor
# output:
# transformed data resolving slow eye movements
def transform_slow(x, f=20):
    return np.tanh(x*f)


# transformation for fast subnet
# input:
#   X: array of shape (N,seqlen,2) where last dimension contains dx, dy gaze velocity components
#   threshold: minimal velocity of saccades in °/ms
# output: transformed data resolving fast eye movements with a velocity higher than threshold
def transform_fast(X, threshold=0.04):
    sac=np.greater(np.hypot(X[:,:,0],X[:,:,1]),threshold)
    sac=np.reshape(sac, newshape=(sac.shape[0], sac.shape[1], 1))
    return np.multiply(sac,X)


# compute z-scores using mean and sd from training data
def normalize(X, mean_train=-9.26532e-05, sd_train=0.060868427):
    return (X-mean_train)/sd_train



# load trained model:
model = load_model('model.json', 'model.h5')
# extract slow subnet
sub1 = Model(inputs=model.input[0], outputs=model.get_layer('s1_sm').output)
# extract fast subnet
sub2 = Model(inputs=model.input[1], outputs=model.get_layer('s2_sm').output)


# load test data: data from 75 readers seeing one text
#   The data is sorted by readers and, within each reader, chronologically.
X = np.load('data.npy')
# X.shape[0]: number of sequences
# X.shape[1]: sequences containing 1000 samples = 1000ms
# X.shape[2]: gaze velocities (horizontal and vertical components)

# load test labels: reader Ids
Y = np.load('labels.npy')

# apply transformations
X1 = transform_slow(X)
X2 = transform_fast(X)
# normalize X2 (z-scores)
X2 = normalize(X2)

num_secs = list(np.array(np.arange(num_secs) + 1,dtype=np.int32))
    
    
# predict
# DeepEyedentification
predictions = model.predict([X1,X2])
# slow subnet
predictions_sub1 = sub1.predict(X1)
# fast subnet
predictions_sub2 = sub2.predict(X2)

unique_user = list(np.unique(Y))
num_for_user = []
for i in range(len(unique_user)):
    num_for_user.append(len(np.where(Y == unique_user[i])[0]))

accs_deep_eye = []
accs_slow = []
accs_fast = []
for window_size in num_secs:
    small_user_ids = np.where(num_for_user < window_size)[0]
    # DeepEyedentification
    tmp_predictions = []
    tmp_label = []
    for i in range(X1.shape[0]-window_size): # make predictions for 1, ..., num_sec seconds of gaze data
        cur_predictions = predictions[i:i+window_size,:]
        if len(np.unique(Y[i:i+window_size])) > 1 \
            or cur_predictions.shape[0] < window_size: 
            continue
        cur_mean_scores = np.mean(cur_predictions,axis=0)
        cur_prediction = np.argmax(cur_mean_scores)
        tmp_predictions.append(cur_prediction)
        tmp_label.append(float(np.unique(Y[i:i+window_size])))
    # process user that have fewer data than window_size
    for i in range(len(small_user_ids)):
        cur_predictions = predictions[Y == small_user_ids[i],:]
        cur_mean_scores = np.mean(cur_predictions,axis=0)
        cur_prediction = np.argmax(cur_mean_scores)
        tmp_predictions.append(cur_prediction)
        tmp_label.append(float(small_user_ids[i]))
    tmp_predictions = np.array(tmp_predictions)
    tmp_label = np.array(tmp_label)
    acc = np.sum(tmp_predictions == tmp_label) / len(tmp_label)
    accs_deep_eye.append(acc)
    
    # slow subnet
    tmp_predictions = []
    tmp_label = []
    for i in range(X1.shape[0]-window_size):
        cur_predictions = predictions_sub1[i:i+window_size,:]
        if len(np.unique(Y[i:i+window_size])) > 1 \
            or cur_predictions.shape[0] < window_size:
            continue
        cur_mean_scores = np.mean(cur_predictions,axis=0)
        cur_prediction = np.argmax(cur_mean_scores)
        tmp_predictions.append(cur_prediction)
        tmp_label.append(float(np.unique(Y[i:i+window_size])))
    # process user that have fewer data than window_size
    for i in range(len(small_user_ids)):
        cur_predictions = predictions_sub1[Y == small_user_ids[i],:]
        cur_mean_scores = np.mean(cur_predictions,axis=0)
        cur_prediction = np.argmax(cur_mean_scores)
        tmp_predictions.append(cur_prediction)
        tmp_label.append(float(small_user_ids[i]))
    tmp_predictions = np.array(tmp_predictions)
    tmp_label = np.array(tmp_label)
    acc = np.sum(tmp_predictions == tmp_label) / len(tmp_label)
    accs_slow.append(acc)
    
    # fast subnet
    tmp_predictions = []
    tmp_label = []
    for i in range(X1.shape[0]-window_size):
        cur_predictions = predictions_sub2[i:i+window_size,:]
        if len(np.unique(Y[i:i+window_size])) > 1 \
            or cur_predictions.shape[0] < window_size:
            continue
        cur_mean_scores = np.mean(cur_predictions,axis=0)
        cur_prediction = np.argmax(cur_mean_scores)
        tmp_predictions.append(cur_prediction)
        tmp_label.append(float(np.unique(Y[i:i+window_size])))
    # process user that have fewer data than window_size
    for i in range(len(small_user_ids)):
        cur_predictions = predictions_sub2[Y == small_user_ids[i],:]
        cur_mean_scores = np.mean(cur_predictions,axis=0)
        cur_prediction = np.argmax(cur_mean_scores)
        tmp_predictions.append(cur_prediction)
        tmp_label.append(float(small_user_ids[i]))
    tmp_predictions = np.array(tmp_predictions)
    tmp_label = np.array(tmp_label)
    acc = np.sum(tmp_predictions == tmp_label) / len(tmp_label)
    accs_fast.append(acc)
        
    

colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
marker = ['-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-','-']
f = plt.figure()
plt.plot(num_secs,accs_deep_eye,label='DeepEyedentification',lw=4,color=colors[2])
plt.plot(num_secs,accs_slow,label='slow subnet',lw=4,color=colors[0])
plt.plot(num_secs,accs_fast,label='fast subnet',lw=4,color=colors[1])

plt.legend(loc=4,fontsize=19)
plt.grid('on')
plt.yticks(fontsize=19)
plt.xticks(num_secs,fontsize=19)
plt.ylabel('Classification accuracy')
plt.xlabel('Time to classification in seconds')
f.set_size_inches((10, 8))
plt.show()
