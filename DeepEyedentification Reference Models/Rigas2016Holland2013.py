# -*- coding: utf-8 -*-
"""
Re-implementation of Rigas et al. 2016 and Holland and Komogortsev, 2013.
Rigas et al. 2016 is an extension of  Holland and Komogortsev, 2013 and 
uses a superset of the features used by Holland and Komogortsev, 2013.

Since not all details are clear from the published papers, I make the following assumptions:
Assumptions:
	- use absolute values for saccade amplitude (not explicitly specified by the authors,
	   but Fig 1 of Rigas et al. 2016 implies that amplitude is positive)
	- use absolute velocities (not explicitly specified by the authors,
	   but Fig 1 of Rigas & Komorgortsev implies that velocity is positive)
	- According to the equations provided by Rigas et al. (2016), saccades must be longer 
       than 16 samples, otherwise, velocity and acceleration-based features cannot be computed;
       I am removing saccades below this threshold
	- it is not defined how the edges of the velocity and accelration vectors are computed; 
       I am using missing values.
    - From the paper, it seems that the authors use screen coordinates for all feature computations
       including saccadic veloctiy and acceleration. I am using gaze coordinates for the latter.
"""


class Screen:  # properties of the screen
    def __init__(self, screenPX_x, screenPX_y,screenCM_x, screenCM_y, dist):
        self.px_x = screenPX_x # screen width in pixels
        self.px_y = screenPX_y
        self.cm_x = screenCM_x # screen width in cm
        self.cm_y = screenCM_y
        # maximal/minimal screen coordinates in degrees of visual angle:
        self.x_max = pix2deg(screenPX_x-1, screenPX_x, screenCM_x, dist)
        self.y_max = pix2deg(screenPX_y-1, screenPX_y, screenCM_y, dist)
        self.x_min = pix2deg(0, screenPX_x, screenCM_x, dist)
        self.y_min = pix2deg(0, screenPX_y, screenCM_y, dist)

class Experiment:
    def __init__(self, screenPX_x, screenPX_y,screenCM_x, screenCM_y, dist, sampling):        
        self.sampling = sampling # sampling rate in Hz
        self.dist = dist      # eye-to-screen distance in cm
        self.screen = Screen(screenPX_x, screenPX_y,screenCM_x, screenCM_y, dist)

def pix2deg(pix, screenPX,screenCM,distanceCM, adjust_origin=True):
  # Converts pixel screen coordinate to degrees of visual angle
  # screenPX is the number of pixels that the monitor has in the horizontal
  # axis (for x coord) or vertical axis (for y coord)
  # screenCM is the width of the monitor in centimeters
  # distanceCM is the distance of the monitor to the retina 
  # pix: screen coordinate in pixels
  # adjust origin: if origin (0,0) of screen coordinates is in the corner of the screen, set adjust_origin=True to center coordinates
  pix=np.array(pix)
  # center screen coordinates such that 0 is center of the screen:
  if adjust_origin: 
      pix = pix-(screenPX-1)/2 
  # eye-to-screen-distance in pixels
  distancePX = distanceCM*(screenPX/screenCM)
  return np.arctan2(pix,distancePX) * 180/np.pi 

def CEMB_fixation_feats(fix, onset_time, expt):
    # fix: array of shape (N,2); choronologically sorted consecutive samples forming one fixation; columns: x and y gaze coordinates 
    # onset_time: timestamp of saccade onset (absolut time from beginning of the trial)
    # sampling_freq: sampling frequency of the eyetracker
    # time_col: column of fix containing time stamp
    # x_col, y_col: columns of fix containing x and y coordinates
    start = onset_time
    dur = fix.shape[0]*(1000/expt.sampling)
    cent_x = np.mean(fix[:,0]) 
    cent_y = np.mean(fix[:,1])
    return start, dur, cent_x, cent_y

def CEMB_saccade_feats(sac, onset_time, expt):
    # sac: array of shape (N,2) choronologically sorted consecutive samples forming one saccade:  col 0,1: x and y gaze coordinates 
    # onset_time: timestamp of saccade onset (absolut time from beginning of the trial)
    start = onset_time
    dur = sac.shape[0]*(1000/expt.sampling)
    ampl_x = np.abs(pix2deg(sac[-1,0],expt.screen.px_x, expt.screen.cm_x, expt.dist) - pix2deg(sac[0,0], expt.screen.px_x, expt.screen.cm_x, expt.dist)) # amplitude in degrees of visual angle
    ampl_y = np.abs(pix2deg(sac[-1,1],expt.screen.px_x, expt.screen.cm_x, expt.dist) - pix2deg(sac[0,1],expt.screen.px_x, expt.screen.cm_x, expt.dist))
    v = np.abs(vel(sac, expt))
    mean_vel_x = np.nanmean(v[:,0])  
    mean_vel_y = np.nanmean(v[:,1])
    max_vel_x = np.nanmax(v[:,0])
    max_vel_y = np.nanmax(v[:,1], )
    return start, dur, ampl_x, ampl_y, mean_vel_x, mean_vel_y, max_vel_x, max_vel_y

def vel(x, expt):
    # compute velocity as in Rigas & Komorgortsev, Eq 2 (x and y velocity-components separately)
    # x: array containing x and y pixel coordinates as columns
    N = x.shape[0]
    # convert screen coordinates into degrees of visual angle
    x[:,0] = pix2deg(x[:,0], expt.screen.px_x, expt.screen.cm_x, expt.dist)
    x[:,1] = pix2deg(x[:,1], expt.screen.px_x, expt.screen.cm_x, expt.dist)
    v = np.zeros((N,2)) # first column for x-velocity, second column for y-velocity
    v[3:N-3, :] = (expt.sampling/6)*(x[0:N-6,:]-x[6:N,:])
    v[0:3] = np.nan # missing values at the edges 
    v[-3:] = np.nan
    return v

def peakvel(ampl, a, b, ampl_min=0.01):   
# Eq. 3 from Rigas et al., 2016 bzw. Baloh et al 1975, p. 1069, right column
# curve to fit relationship between peak velocity and apmplitude of a saccade
# ampl: one-dimensional vector containing absolute amplitude of each saccade (only horizontal or only vertical direction)
# a,b: parameters to be fitted (a: vigor; b: rate)
# ampl_min: minimal amplitude (0 x- or y-amplitudes of purely vertical/horizontal saccades will be set to this value to avoid divide by zero)
    assert np.greater(ampl,0).all() # make sure *absolute* amplitude is given as input and no 0-amplitudes occur (lead to overflow in curve-fitting)
    # in case a saccade is only horizontal or only vertical, x or y ampl can be zero.
    # to avoid divide by zero, set 0s to ampl_min
    peakVel = a*(1-np.exp(-ampl/b))
    return peakVel

# to be applied on each each subject and each saccade of test data
def sac_vigor(sac, b, expt, ampl_min=0.01):
    # b: array of shape (2,) global rate parameters for x and y-direction of saccades (to be estimated from training data)
    # ampl_min: minimal amplitude (0 x- or y-amplitudes will be set to this value to avoid divide by zero)
    peakVel = np.nanmax(np.abs(vel(sac, expt)), axis=0)
    ampl = np.abs(pix2deg(sac[-1], expt.screen.px_x, expt.screen.cm_x, expt.dist)-pix2deg(sac[0], expt.screen.px_x, expt.screen.cm_x, expt.dist)) #x, and y-amplitudes in degrees of visual angle
    if ampl[0]==0: 
        ampl[0]=ampl_min
    if ampl[1]==0:
        ampl[1]=ampl_min
    assert np.greater(ampl,0).all()
    vigor = peakVel/(1-np.exp(-ampl/b)) # x- and y-saccadic vigor (Eq 4 of Rigas et al., 2016)
    return vigor

def acc(v, expt):
    # compute acceleration as in Rigas & Komorgorstev, Eq. 5
    # v: array containing horizontal and vertical velocity as columns
    N = v.shape[0]  
    acc = np.zeros((N,2)) # first column for x-acceleration, second column for y-acceleration
    acc[4:N-4, :] = (expt.sampling/8)*(v[0:N-8,:]-v[8:N,:])
    acc[0:4] = np.nan
    acc[-4:] = np.nan # additional 3 nans at beginning and end introduced above due to nans in velocity
    return acc

def sac_acc_feats(sac, expt):
    v = vel(sac,expt)
    a = acc(v, expt)
    a_x = np.nanmean(a[:,0]) # horizontal mean acceleration
    a_y = np.nanmean(a[:,1]) # vertical mean acceleration
    if np.nanmin(a[:,0]) == 0: # avoid divide by 0
        r_x = np.nan
    else: 
        r_x = np.abs(np.nanmax(a[:,0]))/np.abs(np.nanmin(a[:,0]))  # horizontal peak acceleration ratio (Eq. 6 of Rigas & Komorgortsev, 2016)
    if np.nanmin(a[:,1]) == 0:
        r_y = np.nan
    else:
        r_y = np.abs(np.nanmax(a[:,1]))/np.abs(np.nanmin(a[:,1])) # vertical peak acceleration ratio
    return a_x, a_y, r_x, r_y

def compute_features(dat,  model, expt, b=None,): # extract features from one recording 
    # dat: dataframe containing x,y,start_event (absolute time of event onset from beginning of the trial), event_len (in number of samples, i.e. ms iff 1000Hz), fix (1 for fixation, 0 for saccade, -1 for corrupt event) 
    # b: tuple global rate parameter for saccadic vigor (only used if model='rigas16'); separate values for x and y components of saccades
    # model: 'rigas16' or 'CEM-B'
    # expt: experiment-object
    from ast import literal_eval
    N_fixFeats = 4
    assert model in ['rigas16', 'CEM-B']
    if model=='rigas16':
        assert b[0]
        assert b[1] # b is not None
        N_sacFeats = 14
    else: N_sacFeats = 8
    X_fix = np.zeros(shape=(1,N_fixFeats)) #  fixation features 
    X_sac = np.zeros(shape=(1,N_sacFeats)) # saccade features  
    #dat = dat.reset_index()
    for row_index, row in dat.iterrows(): # iterate over all events (note: row index is NOT consecutive, coming from outer datagrame)
        if row.fix==-1: # ignore corrupt event
            continue
        x = np.array([literal_eval(row.x),literal_eval(row.y)]).transpose() # extract event; literal_eval() because x, y are strings
        onset_x = row.start_time # start time of the event
        if row.fix==1: # if x is fixation event
            X_fix = np.append(X_fix, np.reshape(CEMB_fixation_feats(x, onset_x, expt), newshape=(1,4)), axis=0)
        if row.fix==0 and row.event_len>16: # if saccade of more than 16 samples (skip saccades that have less samples, because they are too short to compute smoothed velocity and acceleration)
            feats = np.zeros(shape=(1,N_sacFeats))
            feats[0,0:8] = CEMB_saccade_feats(x, onset_x, expt) # original features of Holland & Komorgortsev (2013)'s CEM-B model
            if model=='rigas16':
                feats[0,8:10] = sac_vigor(x, b, expt) # additional features of Rigas et al., 2016: saccadic vigor
                feats[0,10:14] = sac_acc_feats(x, expt) # additional features of Rigas et al., 2016: acceleration profile
            X_sac = np.append(X_sac, feats, axis=0)
    if X_fix.shape[0]>1: # if there is at least one fixation event in dat
        X_fix = X_fix[1:] # remove first row which is by construction zeros (keep zeros if no fixation)
    if X_sac.shape[0]>1:
        X_sac =  X_sac[1:]
    return X_fix, X_sac 

# compute saccade amplitude and peak velocity (x, y components separately)
def ampl_peakVel(dat, expt):
    # dat: dataframe; columns: x,y, fix, event_len
    from ast import literal_eval
    X_sac = np.zeros(shape=(1,4)) # 4 saccade features (x,y-amplitude; x,y_peakVel  
    for row_index, row in dat.iterrows(): # iterate over all events
        if row.fix==0 and  row.event_len>16: # if saccade of minimally 17s (otherwise too short to compute smoothed velocity and acceleration)
            sac = np.array([literal_eval(row.x), literal_eval(row.y)]).transpose() # literal_eval beacause x, y are strings;
            feats = np.zeros(shape=(1,4))
            feats[0,0:2] =  np.abs(pix2deg(sac[-1],expt.screen.px_x, expt.screen.cm_x, expt.dist) - pix2deg(sac[0],expt.screen.px_x, expt.screen.cm_x, expt.dist)) # x,y-amplitude in degrees of visual angle
            feats[0,2:4] = np.nanmax(np.abs(vel(sac, expt)), axis=0)
            X_sac = np.append(X_sac, feats, axis=0)
        else: continue # skip fixation or corrupt events
    return X_sac[1:] # first row are zeros by construction

# compute parameter b of peakvel function averaged over all subjects
def global_rate_param(dat, expt): 
    from scipy.optimize import curve_fit
    rate_x = []
    rate_y = []
    for viewer in np.unique(dat.id): # for all viewers
        x = dat[np.equal(dat.id,viewer) ] 
        # compute x/y-amplitude and x/y-peak-velocity of all saccades 
        sacs = ampl_peakVel(x, expt) 
        
        ampl_x = sacs[:,0]
        maxVel_x = sacs[:,2]
        # in case of purely vertical saccades, ampl_x can be 0
        # ---> remove those saccades (from the x-component arrays of amplitudes and maxVel, not from the y-component arrays)
        maxVel_x = maxVel_x[np.greater(ampl_x,0)]
        ampl_x = ampl_x[np.greater(ampl_x,0)]
        
        ampl_y = sacs[:,1]
        maxVel_y = sacs[:,3]
        # remove saccades with zero-vertical-amplitude from y-component data (from amplitudes and maxVel arrays)
        maxVel_y = maxVel_y[np.greater(ampl_y,0)]
        ampl_y = ampl_y[np.greater(ampl_y,0)]
        
        rate_x.append(curve_fit(peakvel, ampl_x, maxVel_x)[0][1])
        rate_y.append(curve_fit(peakvel, ampl_y, maxVel_y)[0][1]) 
    rate = np.array([np.mean(rate_x), np.mean(rate_y)])
    print('global rate param: ', rate)
    return rate 

# train model: compute global rate parameter and compute features
def train_model(dat, model, expt, multiclass=True):
    # dat: pandas dataframe containing training data; chronologically sorted
    Feats = pd.DataFrame(columns=['id', 'fix_feats', 'sac_feats']) 
    viewers = np.unique(dat.id)
    if model=='rigas16':
        # compute global paramters  (Eq. 3 from Frigas & Komorogortsev, 2016 bzw. Baloh et al 1975, p. 1069, right column)
        print('fit global rate param')
        rate = global_rate_param(dat, expt) # rate parameter for x- and y direction
    else: rate=None
    if multiclass: # if model is trained for multiclass classification setting (not needed for identification setting)
        # for each viewer compute features (using all trials)
        for viewer in viewers:
            t = dat[np.equal(dat.id, viewer)].copy()
            # compute features  
            feats = compute_features(t, model, expt, rate)
            f=pd.DataFrame([[viewer, feats[0], feats[1]]], columns=['id', 'fix_feats', 'sac_feats'])
            Feats = pd.concat([Feats, f], axis=0) 
            Feats.reset_index(drop=True, inplace=True)
        return rate, Feats
    else: 
        return rate

def enrol(dat, model, expt):
    # dat: pandas dataframe containing chronologically sorted enrolment data from all to be enrolled users
    Feats = pd.DataFrame(columns=['id', 'fix_feats', 'sac_feats']) 
    viewers = np.unique(dat.id)
    # for each viewer compute features (using all trials)
    for viewer in viewers:
        print('compute enrolment features, viewer: ', viewer)
        t = dat[np.equal(dat.id, viewer)].copy()
        # compute features  
        feats = compute_features(t, model, expt)
        f=pd.DataFrame([[viewer, feats[0], feats[1]]], columns=['id', 'fix_feats', 'sac_feats'])
        Feats = pd.concat([Feats, f], axis=0)            
    Feats.reset_index(drop=True, inplace=True)
    return Feats       

# compute similarity of two trials for all features
def feature_distance_scores(sample1, sample2): 
    # sample1, sample2: tuples of fixations-features and saccade-features arrays: trials/recordings to compare
    from scipy.stats import ks_2samp # 2-sample Kolmogorov-Smirnov test
    assert sample1[0].shape[1] == sample2[0].shape[1] and sample1[1].shape[1] == sample2[1].shape[1] # same number of features
    NfixFeats = sample1[0].shape[1] 
    NsacFeats = sample1[1].shape[1]
    Nfeats = NfixFeats+NsacFeats # total number fixation and saccade features
    similarities = np.zeros(shape=(Nfeats))
    for feat in range(NfixFeats):
        similarities[feat] = ks_2samp(sample1[0][:,feat], sample2[0][:,feat])[0]
    for feat in range(NsacFeats):
        similarities[NfixFeats+feat] = ks_2samp(sample1[1][:,feat], sample2[1][:,feat])[0]
    return similarities # similarities of each feature

def score_fusion(feature_dists, metric='sum'):
    if metric == 'sum':
        return np.sum(feature_dists)

def predict_id(test_sample, train_dat, rate=None, model='rigas16'):
    # test_sample:  dataframe containing data from one test trial/recording (coming from one unique viewer)
    # train_dat: dataframe: each row contains training data from one subject
	# rate parameter for Rigas et al.
    feats_test = compute_features(test_sample, model, expt, rate) 
    dists = np.zeros(shape=(train_dat.shape[0],2))
    for index, row in train_dat.iterrows():
        feats_train = (row.fix_feats,row.sac_feats)
        # similarity between test trial and this training trial
        dists[index,0] = score_fusion(feature_distance_scores(feats_test, feats_train))
        # viewer id of train trial
        dists[index,1] = row.id
    # predicted id
    pred_id = dists[np.argmin(dists[:,0]),1]
    pred_dist = dists[np.argmin(dists[:,0]),0]
    return pred_id, pred_dist

# leave one stimulus out evaluation
def evaluate_loo(d, Nsecs, model, expt, window_stepsize=1000):
    # d: dataframe with raw data (training and test)
    # Nsecs: list containing number of seconds used as input at test time
    accuracies = np.zeros(shape=(len(np.unique(d.img)), len(Nsecs)))
    for index_test_img, test_img in enumerate(np.unique(d.img)): # split over stimuli
        print('test img: ', test_img)
        dat_train = d[~np.equal(d.img, test_img)].copy() # same training data used for varying number of test seconds
        rate, feats_train = train_model(dat_train, model, expt)
        for index_Nsec, Nsec in enumerate(Nsecs):
            #print('starting evaluation for Nsec: ', Nsec)
            Nms = 1000*Nsec # start and end times are in ms
            accs = [] 
            for viewer in np.unique(d.id):
                print('viewer ', viewer)
                # test data from this viewer
                dat_test = d[np.logical_and(np.equal(d.id,viewer), np.equal(d.img,test_img))].copy() 
                accs_window = [] # accuracy for each step of the moving window
                for window_onset in np.arange(0, dat_test.end_time.iloc[-1]-Nms, window_stepsize):    
                    window_offset = window_onset+Nms # offset NOT included!  
                    test_samples = dat_test[np.logical_and(np.greater_equal(dat_test.start_time,window_onset), np.greater(window_offset, dat_test.end_time))].copy() # if not enough data for at least one window, aray is empty --> no problem as long as for each subject from at least one trial there is enough data
                    if not test_samples.empty: # test_samples is empty in case an event is longer than Nms
                        pred_id, pred_dist = predict_id(test_samples.copy(), feats_train, rate, model)
                        accs_window.append(viewer==pred_id)
                print('viewer: ', viewer, 'test_img: ', test_img, ' Nsec: ', Nsec, ' acc: ', np.nanmean(accs_window),'number of samples for window: ', len(accs_window))    
                if accs_window: # if at least 1 window fits into test_samples
                    accs.append(np.mean(accs_window)) # average over moving window
                else: print('For viewer ', viewer, 'img, ', test_img, 'window of Nsec =', Nsec, 'does not fit into data (not enough data)')          
            print('test_img: ', test_img, ' Nsec: ', Nsec, ' acc: ', np.mean(accs))    
            accuracies[index_test_img, index_Nsec] = np.mean(accs) # average over viewers
    return accuracies

# evaluate on one stimulus
def evaluate_img(d, test_img, Nsecs, model, expt, window_stepsize=1000):
    import numpy as np
    # d: dataframe with raw data (training and test)
    # Nsecs: list containing number of seconds used as input at test time
    accuracies = np.zeros(shape=(len(Nsecs)))
    dat_train = d[~np.equal(d.img, test_img)].copy() # same training data used for varying number of test seconds
    rate, feats_train = train_model(dat_train, model, expt)
    print(feats_train.shape)
    # iterate over number of test seconds
    for index_Nsec, Nsec in enumerate(Nsecs):
        print('starting evaluation for Nsec: ', Nsec)
        Nms = 1000*Nsec # start and end times are in ms
        accs = [] # accuracy for each reader, averaged over windows of the same size
        for viewer in np.unique(d.id):
            # test data from this viewer
            dat_test = d[np.logical_and(np.equal(d.id,viewer), np.equal(d.img,test_img))].copy()
            accs_window = [] 
            for window_onset in np.arange(0, dat_test.end_time.iloc[-1]-Nms, window_stepsize):    
                window_offset = window_onset+Nms 
                test_samples = dat_test[np.logical_and(np.greater_equal(dat_test.start_time,window_onset), np.greater(window_offset, dat_test.end_time))].copy()
                if not test_samples.empty: # test_samples is empty in case an event is longer than Nms
                    pred_id, pred_dist = predict_id(test_samples, feats_train, rate, model)
                    accs_window.append(viewer==pred_id)    
            if accs_window: # if at least 1 window fits into test_samples
                accs.append(np.mean(accs_window)) # average over moving window
                print('viewer: ', viewer, 'test_img: ', test_img, ' Nsec: ', Nsec, ' acc: ', np.mean(accs_window), 'number of samples for window: ', len(accs_window))    
            else: print('For viewer ', viewer, 'img, ', test_img, 'window of Nsec =', Nsec, 'does not fit into data (not enough data)')
        print('test_img: ', test_img, ' Nsec: ', Nsec, ' acc: ', np.mean(accs))    
        accuracies[index_Nsec] = np.mean(accs) # average over viewers
    return accuracies
