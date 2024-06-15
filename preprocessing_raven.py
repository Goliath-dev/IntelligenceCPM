# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 22:03:52 2023

@author: Admin
"""

# This script preprocesses the test data taken from laboratory (roughly, "Raven" data).

import mne
import intel_utils
import os
import glob
import autoreject
import source_utils
import numpy as np
from mne_connectivity import spectral_connectivity_epochs

# Path to raw data.
raw_path = 'D:\EEG_data\Connectivity\CLEAR_DATA\EEG'
# Creating a list of files in raw_path directory and a list of IDs.
files = glob.glob(raw_path + '\\*con*_f_*.vhdr')
subjects = [file.split('\\')[-1].split('.')[0] for file in files]
subjects = [subj.replace('_f_', '') for subj in subjects]

# Output directories.
conn_dir = 'Matrices'
labels_dir = 'Labels'
source_ts_dir = 'Source time series'
clean_epochs_dir = 'Preprocessed data'

conn_methods = ['wPLI', 'imCoh', 'PLV']
atlas = 'Destrieux'

# Frequency bands.
fmin = (4, 4,  8,  8,  10, 13, 20, 30, 30, 4)
fmax = (8, 30, 13, 10, 13, 20, 30, 40, 45, 45)

# Time points where eyes were open.
open_eyes_times = [[120, 240], [360, 480]]

# A set of participants data. See the file to clarify what's inside.
dfml_csv = intel_utils.read_csv('Raven data\\df_ML_full.csv')
# Take IDs from it to check which subjects is in.
ids = set(dfml_csv[1:, 0])
# Iterate through raw files.
for subject, file in zip(subjects, files):
    # Skip if already preprocessed.
    # if os.path.exists(f'{clean_epochs_dir}/{subject}_clean_epo.fif'): continue
    # Skip if there's no participant data available.
    if subject not in ids: continue

    # Read the data.
    raw = mne.io.read_raw_brainvision(file, preload=True)
    # Skip if the data is too short.
    if raw.times[-1] < 600: continue
    # If there's no locations, set the standard montage.
    if np.isnan(raw.info['chs'][0]['loc'][0]): raw.set_montage('standard_1020')
    # Change the discretization frequency in order to shorten the preprocessing time.
    # Do remember that this value must be at least 2 times more than the largest 
    # frequency of interest.
    raw.resample(200.0)
    
    # Create a list of open eyes epochs...
    open_eyes_epochs_list = []
    for times in open_eyes_times:
        raw_rest = raw.copy().crop(tmin=times[0], tmax=times[1])
        raw_rest.set_eeg_reference('average', projection=True)
        raw_rest.apply_proj()
        raw_rest.filter(1, 45)
        
        oe_epochs = mne.make_fixed_length_epochs(raw_rest, duration=6.,
                                                 preload=True, overlap=1.)
        open_eyes_epochs_list.append(oe_epochs)
    
    # ... and apply this list to glue the separate epochs instances into one.
    epochs = mne.concatenate_epochs(open_eyes_epochs_list, add_offset=False)
    # Set up the rejection threshold. Used before, but not now.
    # reject = autoreject.get_rejection_threshold(epochs, ch_types = ['eeg'], 
    #                                         random_state=42)
    # Set up the ICA in order to perform the ocular and muscular correction.
    n_components = epochs.info['nchan']-len(epochs.info['bads'])- 10
    ica = mne.preprocessing.ICA(n_components = n_components, method = 'infomax',
                                random_state=42)
    ica.fit(epochs)
    # The corrections themselves.
    eog_channel = 'Fp2'  
    eog_inds, eog_scores = ica.find_bads_eog(epochs, ch_name = eog_channel, 
                                    measure = 'correlation', threshold = 0.4)
    muscle_inds, muscle_scores = ica.find_bads_muscle(epochs)
    # ECG correction if possible.
    if 'ecg' in epochs.get_channel_types():
        ecg_inds, ecg_scores = ica.find_bads_ecg(epochs)
        ica.exclude.extend(ecg_inds)
    # Exclude the ICA channels referring to ocular/muscular artifacts and fix the epochs.
    ica.exclude.extend(eog_inds)
    ica.exclude.extend(muscle_inds)
    ica.apply(epochs)
    
    # Set up and run the Autoreject.
    ar = autoreject.AutoReject(random_state=42)
    clean_epochs, rej_log = ar.fit_transform(epochs, return_log=True)
    
    rsc = autoreject.Ransac(verbose = False, n_resample=100, min_channels=0.2, min_corr=0.9, n_jobs = -1, random_state=42)
    clean_epochs = rsc.fit_transform(clean_epochs)
    
    # A dirty trick to prevent source rocenstruction from crash because of wrong 
    # reference projector due to removing bads in autorject.
    clean_epochs.set_eeg_reference('average', projection=True)
    clean_epochs.apply_proj()
    
    # A number of interpolated channels.
    n_inter_chls = len(rsc.bad_chs_)
    
    # Solve the inverse problem and extract the time series of the sources.
    # Also extract the label objects.
    labels, ts = source_utils.fsaverage_time_courses(clean_epochs, clean_epochs.info, method='eLORETA', parc = 'aparc.a2009s')
    
    # Compute the conncetivity matrices, given method and frequency bands.
    for method in conn_methods:
        # Compute the conncetivity matrices, given method and frequency bands.
        conn = spectral_connectivity_epochs(ts, method=method.lower(), sfreq=epochs.info['sfreq'], 
                                            fmin=fmin, fmax=fmax, 
                                            faverage=True, n_jobs=-1)
        os.makedirs(f'{conn_dir}\\{method}\\{atlas}', exist_ok=True)
        np.save(f'{conn_dir}\\{method}\\{atlas}\\{subject}_ninters_{n_inter_chls}', np.swapaxes(conn.get_data('dense'), 0, 1))
    # Save whatever have.
    # This one GOTTA be one and the same for all subjects, but for God's sake I save 'em all
    os.makedirs(f'{labels_dir}\\{atlas}', exist_ok=True)
    os.makedirs(f'{source_ts_dir}\\{atlas}', exist_ok=True)
    os.makedirs(f'{clean_epochs_dir}/{atlas}', exist_ok=True)
    np.save(f'{labels_dir}/{atlas}/{subject}_labels', labels) 
    np.save(f'{source_ts_dir}/{atlas}/{subject}_source_ts', ts) 
    clean_epochs.save(f'{clean_epochs_dir}/{atlas}/{subject}_clean_epo.fif', fmt='double', overwrite=True)
    
    
    
    
    