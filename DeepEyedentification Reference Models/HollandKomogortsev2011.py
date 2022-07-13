# -*- coding: utf-8 -*-
'''
Re-implementation of Holland & Komogortsev, Biometric identification via eye movement scanpaths in reading, 2011.

Scan-path features:
    - fixation count
    - average fixation duration
    - average vectorial saccade amplitude
    - average horizontal saccade amplitude
    - average vertical saccade amplitude
    - average vecotrial saccade velocity
    - average vecotrial saccade peak velocity
    - velocity waveform indicator
    - scanpath length
    - scanpath area
    - regions of interest
    - inflection count
    - slope coefficient of the amplitude-duration relationship
    - slope coefficient of the main sequence relationship


Since some implementational details are not specified in the paper, the following additional 
assumptions are made:
	- The authors state to use a velocity-based algorithm to preprocess the data into saccades
		and fixations; but it is unclear with specific instantiation of this class of algorithms 
		they use. I use the velocity-based algorithm described in 
		Engbert, R. (2006). Microsaccades: A microcosm for research on oculomotor control, 
		attention, and visual perception. Progress in Brain Research, 154, 177–192.

	- The authors do not say how they compute the gaze velocity; I am using the same 
		moving-average-smoothing algorithm as for the preprocessing of the data
	- The authors do not state whether they use absolute values for the amplitudes;
		I am using absolute vertical/horizontal amplitudes (makes no sense otherwise because of averaging)
    - Average vectorial saccade velocity (see page 3 of Holland and Komogortsev, 2011) is not well defined:
		"the vectorial velocity of a saccade was defined as the Eucildean norm of the horizontal and vertical velocities" 
		 ---> I am assuming the authors mean the "horizontal and vertical *mean* velocity".
	- The authors define the average vectorial saccade peak velocity as 
		"the vecotrial peak velocity of a saccade was defined as the Eucildean norm of the horizontal 
		and vertical peak velocities." (page 3)
		 ---> it is unclear whether the authors assume that the horizontal and vertical peak velocity is necessarily reached at the same time.
			This assumptions is not true in the data. I am therefore defining the vectorial peak velocity as the maximal vectorial velocity.
	- Scanpath length (page 4 of Holland and Komogortsev, 2011): Equation 2 does not correspond to the definition 
		of scanpath length in the reference cited by the authors. I am using the definition from the cited reference assuming that Eq. 2 is a mistake.
    - to identify the regions of interest, fixation locations are needed; I use the mean of the x-coordinates and the mean of the y-coordinates as location
    - for the computation of the regions of interest, I use the mean shift clustering algorithm referenced by the authors using the same parameters as them.
		Optimally,  the sigma parameter should be tuned or selected based on the stimulus; since the authors do not report how they arrived at 
		their value for sigma, I simply use theirs. Sigma influences the spatial resolution of the clustering.
    - Pairwise distance comparison feature: I do not use this feature since it does not seem useful when using different stimuli for training and testing.
    - the original model is not evaluated in a multi-class classification setting, but only in a verification setting. In order to apply the model in the multi-class
		classification setting, I compute the features of each scanpath of a subject in the training data and then average the features over all scanpaths from this 
		subject ---> this results in one training feature vector for each subject. The test scanpath is then compared to each subject's training feature vector and the 
		 subject whith the highest similarity to to the test scanpath is predicted; Averaging the training scan path features yields better results than 
		 comparing the test scanpath to all separate training scanpaths.
'''

import numpy as np

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
  # adjust origin: if origin (0,0) of screen coordinates is in the corner of the screen rather than in the center, set to True to center coordinates
  pix=np.array(pix)
  # center screen coordinates such that 0 is center of the screen:
  if adjust_origin: 
      pix = pix-(screenPX-1)/2 # pixel coordinates start with (0,0) 
  # eye-to-screen-distance in pixels
  distancePX = distanceCM*(screenPX/screenCM)
  return np.arctan2(pix,distancePX) * 180/np.pi #  *180/pi wandelt bogenmass in grad 

# Compute velocity times series from 2D position data
# adapted from Engbert et al.  Microsaccade Toolbox 0.9
# x: array of shape (N,2) (x und y screen or visual angle coordinates of N samples in *chronological* order)
# returns velocity in deg/sec or pix/sec
def vecvel(x, sampling_rate=1000, smooth=True):
    N = x.shape[0]  
    v = np.zeros((N,2)) # first column for x-velocity, second column for y-velocity
    if smooth: # v based on mean of preceding 2 samples and mean of following 2 samples
        v[2:N-2, :] =  (sampling_rate/6)*(x[4:N,:] + x[3:N-1,:] - x[1:N-3,:] - x[0:N-4,:])
        v[1,:] = (sampling_rate/2)*(x[2,:] - x[0,:])
        v[N-2,:] = (sampling_rate/2)*(x[N-1,:] - x[N-3,:])
    else:  
        v[1:N-1,] = (sampling_rate/2)*(x[2:N,:] - x[0:N-2,:]) 
    return v

# mean shift cluster algorithm used to compute regions of interest
# Santella Doug DeCarlo (2004): Robust Clustering of Eye Movement Recordings for Quantification of Visual Interest
def k_spatial(x, sigma):
    # x: arrays of shape (2) containing x and y coordinates of preprocessed fixation location
    return np.exp(-(np.power(x[0],2)+np.power(x[1],2))/sigma**2)

def shift(x, sigma):
    # x: arrays of shape (N,2) containing x and y coordinates of preprocessed fixation locations
    x_new=np.zeros(shape=x.shape)
    for i in range(x.shape[0]): # for each fixation
        sims=[]
        normalizer=[]
        # adjust mean
        for j in range(x.shape[0]):
            sims.append(k_spatial(x[i]-x[j],sigma=sigma)*x[j])
            normalizer.append(k_spatial(x[i]-x[j], sigma=sigma))
        x_new[i] = np.sum(sims, axis=0)/np.sum(normalizer, axis=0)
    return x_new

def mean_shift_cluster(x, sigma=2, epsilon=0.1):
    # epsilon: convergence rate
    x_old = np.full(fill_value=np.inf, shape=x.shape)
    while np.greater(np.abs(x_old-x), epsilon).any():
        x_old = x
        x = shift(x, sigma=2)
        return x
 
# compute scan-path area using convex hull Goldberg & Kotval, 1999
def triangle_area(a,b,c):
    # a, b, c: points of the triangle
    p=perimeter(a,b,c) # half perimeter of the triangle
    return np.sqrt(p*(p-dist(a,b))*(p-dist(b,c))*(p-dist(c,a)))

def dist(a,b):
    # a,b: points in 2D space
    return np.sqrt(np.power(a[0]-b[0],2)+np.power(a[1]-b[1],2))

def perimeter(a,b,c):
    return (dist(a,b)+dist(b,c)+dist(c,a))/2.0

def convex_hull_area(x):
    # x: array of shape (N,2) containing x,y locations of fixations
    from scipy.spatial import ConvexHull
    from scipy.spatial.qhull import QhullError
    # vertices of the complex hull:
    try:
        vertices = x[ConvexHull(x).vertices] 
    except QhullError: # not enough points (at least 3) to construct convex hull
        return 0
    subareas=[]
    for n in range(1,vertices.shape[0]-1):
        subareas.append(triangle_area(vertices[0], vertices[n], vertices[n+1]))
    area = np.sum(subareas)
    return area

def scanpath_features(dat): # extract features from one scan-path
    # dat: pandas dataframe containing one scan path: one row per event
    from ast import literal_eval
    from scipy.stats import linregress
    N = 14 # n features
    X = np.zeros(shape=(N)) # features  
    fix_count = 0
    fix_dur =[]
    ampl_x = []
    ampl_y = []
    ampl = []
    mean_vel = []
    peak_vel = []
    peak_vel_x = []    
    peak_vel_y = []
    waveform = []
    dir_x = []
    dir_y = []
    loc_x = []
    loc_y = []
    sac_dur = []
    for row_index, row in dat.iterrows(): # iterate over all events (note: row index is NOT continuous)
        if row.fix==-1: continue # ignore corrupt event
        x = np.array([literal_eval(row.x),literal_eval(row.y)]).transpose() # extract event;
        if row.fix==1: # if row is fixation
            # fixation features
            fix_count = fix_count +1
            fix_dur.append(row.event_len)
            # fixation location (used for computation of regions of interest)
            loc_x.append(np.mean(x[:,0]))
            loc_y.append(np.mean(x[:,1]))
        if row.fix==0: # if saccade
            # saccade features
            # horizontal and vertical amplitude
            ampl_x.append(np.abs(x[0,0] - x[-1,0]))
            ampl_y.append(np.abs(x[0,1] - x[-1,1]))
            # vectorial amplitude
            ampl.append(np.sqrt(np.power(x[0,1]-x[-1,1],2) + np.power(x[0,0]-x[-1,0],2)))
            # average vectorial saccade velocity
            v = vecvel(x) # horizontal and vertical velocity components
            vec_vel = np.hypot(v[:,0], v[:,1]) # vectorial velocity
            mean_vel.append(np.mean(vec_vel))
            # vecotrial saccade peak velocity
            peak_vel.append(np.max(vec_vel))
            # x,y peak velocity (used for main sequence coefficient)
            peak_vel_x.append(np.max(v[:,0]))
            peak_vel_y.append(np.max(v[:,1]))
            # velocity waveform indicator
            waveform.append(np.max(vec_vel)/np.mean(vec_vel))
            # saccade direction (used for inflection count)
            dir_x.append(np.sign(x[-1,0]-x[0,0]))
            dir_y.append(np.sign(x[0,1]-x[-1,1]))
            sac_dur.append(row.event_len)
    # aggregate features:
    X[0]=fix_count
    X[1]=np.mean(fix_dur)
    # only use saccades with horizontal amplitude >0.5°
    X[2]=np.mean(np.array(ampl_x)[np.greater(ampl_x,0.5)])
    # only use saccades with vertical amplitude >0.5°
    X[3]=np.mean(np.array(ampl_y)[np.greater(ampl_y, 0.5)])
    X[4]=np.mean(ampl)
    X[5]=np.mean(mean_vel)
    X[6]=np.mean(peak_vel)
    X[7]=np.mean(waveform)
    # scanpath features
    # scanpath length
    X[8] = np.sqrt(np.power(loc_x[-1] - loc_x[0], 2) +  np.power(loc_y[-1] - loc_y[0], 2))
    # scanpath area, Goldberg & Kotval, 1999
    fix_loc = np.array([loc_x,loc_y]).transpose()
    X[9] = convex_hull_area(fix_loc)
    # number of regions of interest via mean shift clustering by Santella & DeCarlo, 2004
    X[10] = len(np.unique(np.round(mean_shift_cluster(np.array([loc_x, loc_y]).transpose(), sigma=2, epsilon=1),decimals=1), axis=0)) # round to 1 decimal, in line convergence rate epsilon
    # inflection count
    no_inflections_x = np.equal(dir_x[:-1],dir_x[1:])
    no_inflections_y = np.equal(dir_y[:-1],dir_y[1:])
    X[11] = np.sum(~(no_inflections_x*no_inflections_y))
    # slope coefficient for the amplitude-duration relationship
    X[12] = linregress(np.max(np.array([ampl_x,ampl_y]), axis=0), sac_dur)[0]
    # slope coefficient of the main sequence relationship
    X[13] = linregress(ampl_x+ampl_y, peak_vel_x+peak_vel_y)[0]
    return X

# train model: 
# return features for each participant: compute features for each scanpath, then average them ---> representation of this subject
# get estimates for metric-specific standard deviations
def fit_model(d):
    # dat_ dataframe containing raw training data; choronoligcally sorted
    X = np.zeros(shape=(len(np.unique(d.id)), 14)) # viewer ID and features (one row per subject)
    S = np.zeros(shape=(len(np.unique(d.id)), 14)) # within-subject metric specific standard deviations
    viewer_ids = np.unique(d.id)
    for viewer_index, viewer in enumerate(viewer_ids):
        d_viewer = d[np.equal(d.id,viewer)]
        # assumption: each subject saw each image only once 
        scanpath_ids = np.unique(d_viewer.img)
        x=np.zeros(shape=(len(scanpath_ids),14))
        for index, i in enumerate(scanpath_ids):
            #scanpath = d_viewer[np.equal(list(zip(d_viewer.img, d_viewer.sess)), i).all(axis=1)]
            scanpath  = d_viewer[np.equal(d_viewer.img,i)]
            x[index] = scanpath_features(scanpath)  
        X[viewer_index,:] =  x.mean(axis=0) # average features over all scanpaths of a viewer
        S[viewer_index, :] = x.std(axis=0)
    sigmas = 2*S.mean(axis=0) # Equation 8 of H&K
    return X, viewer_ids, sigmas # viewer_id does NOT correspond to row index in X!

def feature_similarity(a,b, sigma):
    # a, b: features to compare 
    # similarity of one feature
    # sigma obtained from training data, see H&K, page 5, Eq. 8
    from scipy.stats import norm
    p = norm.cdf(x=a, loc=b, scale=sigma)  
    return  1 - np.abs(2*p-1)

def similarity_fusion(feature_sims, weights=None):
    # feature_sims: array of shape (14,) containing the similarity between features of 2 given scanpaths
    # weights: array of shape (14,) containing feature weights; 'HK': use feature weights from Table 1 of Holland and Komogortsev, 2011
    if not weights: 
        return np.mean(feature_sims)
    if weights=='HK':
        weights = np.array([25, 75, 10, 1, 1, 10, 1, 1, 1, 10, 1, 10, 1, 1]) # weights from Holland and Komogortsev, 2011
    return np.dot(weights,feature_sims)/np.sum(weights) 

def scanpath_similarity(a,b, sigmas, weights=None):
    # a,b: feature-arrays of two scanpaths of shape (14,)
    # sigmas: array of shape (14,) containing feature specific standard deviation
    # weighting: use feature weights from Table 1 of Holland and Komogortsev, 2011
    assert a.shape==b.shape
    nfeats = a.shape[0]
    sims=np.zeros(shape=(nfeats))
    for feat in range(0, nfeats):
        sims[feat] = feature_similarity(a[feat], b[feat], sigmas[feat])
    sim = similarity_fusion(sims, weights=weights)
    return sim

def predict(Feats_train, sigmas, viewer_ids_train, feats_test):
    # Feats_train: averaged scanpath features from training data of each subject
    # sigmas: metric-specific std, computed from training data
    # viewer_ids_train: array with original viewer ids, sorting corresponds to Feats_train
    # feats_test: array of shape (14,) containing test scanpath
    assert Feats_train.shape[0]==len(viewer_ids_train)
    sims = np.zeros(shape=(Feats_train.shape[0]))
    for viewer_index in range(Feats_train.shape[0]): # for the averaged scanpath-representation of each viewer
        sims[viewer_index] = scanpath_similarity(Feats_train[viewer_index],feats_test,sigmas)
    return viewer_ids_train[np.argmax(sims)]    

# evaluate on one test image
def evaluate_img(d, test_img, Nsecs, expt, window_stepsize=1000):
    import numpy as np
    # d: dataframe with raw data (training and test)
    #Nsecs: list containing number of seconds used as input at test time
    accuracies = np.zeros(shape=(len(Nsecs)))
    dat_train = d[~np.equal(d.img, test_img)].copy() # same training data used for varying number of test seconds
    print('train model')
    Feats_train, viewer_ids_train, sigmas = fit_model(dat_train)
    # iterate over number of test seconds
    for index_Nsec, Nsec in enumerate(Nsecs):
        print('starting evaluation for Nsec: ', Nsec)
        Nms = 1000*Nsec # start and end times are in ms
        accs = [] # accuracy for each reader, averaged over windows of the same size
        for viewer in np.unique(d.id):
            # test data from this viewer
            dat_test = d[np.logical_and(np.equal(d.id,viewer), np.equal(d.img,test_img))].copy()
            accs_window = [] # accuracy for each step of the moving window
            for window_onset in np.arange(0, dat_test.end_time.iloc[-1]-Nms, window_stepsize):    
                window_offset = window_onset+Nms # offset NOT included!  
                test_samples = dat_test[np.logical_and(np.greater_equal(dat_test.start_time,window_onset), np.greater(window_offset, dat_test.end_time))].copy() # if not enough data for at least one window, aray is empty --> no problem as long as for each subject from at least one trial there is enough data
                if not test_samples.empty: # test_samples is empty in case an event is longer than Nms; skip those
                    # skip test samples that do not have at least one fixation and one saccade
                    if np.sum(np.equal(test_samples.fix,1)) > 0 and np.sum(np.equal(test_samples.fix,0)) > 0:
                        feats_test = fit_model(test_samples)[0][0]
                        pred_id = predict(Feats_train, sigmas, viewer_ids_train, feats_test)
                        accs_window.append(viewer==pred_id)    
            if accs_window: # if at least 1 window fits into test_samples
                accs.append(np.mean(accs_window)) # average over moving window
                print('viewer: ', viewer, 'test_img: ', test_img, ' Nsec: ', Nsec, ' acc: ', np.mean(accs_window), 'number of samples for window: ', len(accs_window))    
            else: print('For viewer ', viewer, 'img, ', test_img, 'window of Nsec =', Nsec, 'does not fit into data (not enough data)')
        print('test_img: ', test_img, ' Nsec: ', Nsec, ' acc: ', np.mean(accs))    
        accuracies[index_Nsec] = np.mean(accs) # average over viewers
    return accuracies