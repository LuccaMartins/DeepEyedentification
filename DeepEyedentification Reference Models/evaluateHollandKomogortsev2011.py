import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import sys
import Holland as hl

hl.expt = Experiment(screenPX_x = 1680, screenPX_y = 1050, screenCM_x = 47.5, screenCM_y = 30, dist = 61, sampling=1000)

d = pd.read_csv(os.path.join('data','dat_events.csv', sep='\t')
# 'event':  fix=1, saccade=2, corrupt=3
d['event'] = -np.sign(d.event-2) # code fixation as 1, saccade as 0 and corrupt as -1
d.rename(columns={'seq_x':'x', 'seq_y': 'y', 'reader': 'id',  'text':'img', 'event':'fix'}, inplace=True)
# data contains corrupt events --> will be removed for the computation of features, but considered for evaluation (amount of input data)

np.random.seed(13)
test_imgs = np.unique(d.img)
Njobs = len(test_imgs)

if len(sys.argv) > 1:
    Nsecs = []
    for element in sys.argv[1:]:
        Nsecs.append(int(element))
else:
    Nsecs=[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
N = ''.join('_'+str(i) for i in Nsecs)

acc = Parallel(n_jobs=Njobs)(
            delayed(hl.evaluate_img)(d, test_img, Nsecs, expt) for test_img in test_imgs)
