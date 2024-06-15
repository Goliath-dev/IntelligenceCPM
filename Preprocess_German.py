# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:38:11 2023

@author: Admin
"""

import glob
import mne
import numpy as np
from mne_connectivity import spectral_connectivity_epochs
from mne import compute_covariance, setup_source_space, make_forward_solution 
from mne import read_labels_from_annot, extract_label_time_course
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.datasets import fetch_fsaverage
import os
import os.path as op
import autoreject

conn_dir = 'E:\\Work\\MBS\\EEG\\Modelling\\Intelligence\\Matrices\\'
data_dir = 'D:\EEG_data\German dataset\MPI-Leipzig_Mind-Brain-Body-LEMON\EEG_MPILMBB_LEMON\EEG_Raw_BIDS_ID'
# data_dir = 'F:\EEG_data\German dataset\MPI-Leipzig_Mind-Brain-Body-LEMON\EEG_MPILMBB_LEMON\EEG_Preprocessed_BIDS_ID\EEG_Preprocessed'
clean_epochs_dir = 'E:\\Work\\MBS\\EEG\\Intelligence\Depression\\Epochs'
labels_dir = 'E:\\Work\\MBS\\EEG\\Modelling\\Intelligence\\Labels'
source_ts_dir = 'E:\\Work\\MBS\\EEG\\Modelling\\Intelligence\\Source time series'

fmin = (4, 8,  10, 13, 20, 30)
fmax = (8, 10, 13, 20, 30, 45)

parcs = ['aparc.a2009s', 'aparc']

methods = ['wPLI', 'PLV', 'imCoh']

files = glob.glob(f'{data_dir}\\sub-*\\RSEEG\\sub-*.vhdr')
# files = glob.glob(f'{data_dir}\\sub-*.set')

for file in files:
    subject = file.split('\\')[-1].split('.')[0].split('_')[0]
    # Skip if already computed.
    if op.exists(f'{conn_dir}/imCoh/DK/{subject}.npy'): 
        print('Already computed, skipped.')
        continue
    # data = mne.io.read_raw_eeglab(file, preload=True)
    data = mne.io.read_raw_brainvision(file, preload=True)
    if 'VEOG' in data.info['ch_names']: data.set_channel_types({'VEOG': 'eog'})
    events = mne.events_from_annotations(data)
    time_bounds = []
    
    # Events 1 and 210 mark eye state switch and closed-eyes state, respectfully.
    start, end = np.nan, np.nan
    fragment_finished = False
    for i, event in enumerate(events[0]):
        # If current event is not an eye state switch, we're not interesting in it at all.
        if event[2] != 1: continue
        # In case we're operating with events marked with '1', we're guaranteed
        # to face them only in the middle of the data, so no additional boundary
        # checks needed. I hope so, at least. 
        # Broken hope :-(
        if i + 1 < len(events[0]): 
            next_event = events[0][i + 1]
        else:
            # Three NaNs better than one. Also helps when calling next_event[2].
            next_event = [np.nan, np.nan, np.nan]
        prev_event = events[0][i - 1]
        # Technically, the 'event[2] == 1' condition is superfluous and may be omitted, 
        # but hey.
        # If this is an eye state switch and next event indicates eyes-closed state,
        # then this is a beginning of an eyes-closed part of the record.
        if event[2] == 1 and next_event[2] == 210:
            start = data.times[event[0]]
            fragment_finished = False
        # If this is an eye state switch and previous event indicates eyes-closed 
        # state, then this is an end of an eyes-closed part of the record.
        if event[2] == 1 and prev_event[2] == 210:
            end = data.times[event[0]]
            fragment_finished = True
        if not (np.isnan(start)) and not(np.isnan(end)) and fragment_finished:
            time_bounds.append((start, end))
            fragment_finished = False
     
    # Skip if time_bounds not set due to inconsistent event labeling.
    if len(time_bounds) == 0: continue
    # boundary_events = events[0][events[0][:,2] == 4]
    # time_bounds = np.concatenate([[0], data.times[boundary_events[:,0]], [data.times[-1]]])
    epoch_arr = []
    for bounds in time_bounds:
    # for i in range(len(time_bounds) - 1): 
        # Done with < 3.10 version of Python, so no itertools.pairwise allowed
        # Data is actually discontinued, so retain one second from both bounds
        # to prevent boundary artifacts.
        # left_bound = time_bounds[i] + 1
        # right_bound = time_bounds[i + 1] - 1
        left_bound = bounds[0] + 1
        right_bound = bounds[1] - 1
        # Boundary markers are sometimes set near the beginning and sometimes not,
        # so fix for that case.
        # if right_bound - left_bound < 30: continue
        cropped_data = data.copy().crop(left_bound, right_bound)
        cropped_data.filter(1, 45)
        cropped_data.set_eeg_reference('average', projection=True)
        cropped_data.apply_proj()
        epochs = mne.make_fixed_length_epochs(cropped_data, duration=6.0, overlap=1.0)
        epoch_arr.append(epochs)
    epochs = mne.concatenate_epochs(epoch_arr)
    epochs.resample(200.0)
    epochs.set_montage('standard_1020')
    
    # ICA correction.
    n_components = epochs.info['nchan']-len(epochs.info['bads'])- 10
    ica = mne.preprocessing.ICA(n_components = n_components, method = 'infomax',
                                random_state=42)
    ica.fit(epochs)
    eog_channel = 'Fp2'  
    eog_inds, eog_scores = ica.find_bads_eog(epochs, ch_name = eog_channel, 
                                    measure = 'correlation', threshold = 0.4)
    muscle_inds, muscle_scores = ica.find_bads_muscle(epochs)
    ica.exclude.extend(eog_inds)
    ica.exclude.extend(muscle_inds)
    ica.apply(epochs)
    
    # Set up and run the Autoreject.
    ar = autoreject.AutoReject(random_state=42)
    clean_epochs, rej_log = ar.fit_transform(epochs, return_log=True)
    
    rsc = autoreject.Ransac(verbose = False, n_resample=100, min_channels=0.2, 
                            min_corr=0.9, n_jobs = -1, random_state=42)
    clean_epochs = rsc.fit_transform(clean_epochs)
    
    os.makedirs(f'{clean_epochs_dir}', exist_ok=True)
    clean_epochs.save(f'{clean_epochs_dir}/{subject}_clean_epo.fif', fmt='double', 
                      overwrite=True)
    
    # Source reconstruction.
    noise_cov = compute_covariance(clean_epochs, tmax=0., method=['shrunk', 'empirical'])
    fs_dir = fetch_fsaverage()
    subjects_dir = op.dirname(fs_dir)
    fs_subject = 'fsaverage'
    trans = 'fsaverage'
    src = setup_source_space(subject=fs_subject, subjects_dir=subjects_dir, n_jobs=-1)
    bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    fwd = make_forward_solution(epochs.info, trans=trans, src=src, 
                                    eeg=True, bem=bem, mindist=5.0, n_jobs=-1)
    inverse = make_inverse_operator(epochs.info, fwd, noise_cov, loose=0.2, depth=0.8)
    snr = 3
    lambda2 = 1. / snr ** 2
    stc = apply_inverse_epochs(epochs, inverse, lambda2, method='eLORETA')
    for parc in parcs:
        atlas = 'Destrieux' if parc == 'aparc.a2009s' else 'DK'
        ROI_offset = 2 if parc == 'aparc.a2009s' else 1
        labels = read_labels_from_annot(subject=fs_subject, parc = parc, subjects_dir=subjects_dir)
        time_courses = extract_label_time_course(stc, labels[:-ROI_offset], src)
        os.makedirs(f'{labels_dir}\\{atlas}', exist_ok=True)
        os.makedirs(f'{source_ts_dir}\\{atlas}', exist_ok=True)
        np.save(f'{labels_dir}/{atlas}/{subject}_labels', labels) 
        np.save(f'{source_ts_dir}/{atlas}/{subject}_source_ts', time_courses) 
        for method in methods:
            conn = spectral_connectivity_epochs(time_courses, method=method.lower(), sfreq=epochs.info['sfreq'], 
                                                fmin=fmin, fmax=fmax, 
                                                faverage=True, n_jobs=-1)
            os.makedirs(f'{conn_dir}/{method}/{atlas}/', exist_ok=True)
            np.save(f'{conn_dir}/{method}/{atlas}/{subject}', np.swapaxes(conn.get_data('dense'), 0, 1))
    