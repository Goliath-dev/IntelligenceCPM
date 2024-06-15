# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 22:01:52 2023

@author: Admin
"""

import mne
import intel_utils
import matplotlib.pyplot as plt
import pandas as pd
import mne_bids
import os
from pathlib import Path
import glob
import autoreject
import source_utils
import numpy as np
from mne_connectivity import spectral_connectivity_epochs

behav = intel_utils.read_csv('WAIS_III.csv')
PIQ_with_ID = behav[2:, [0, 3]]
PIQ_with_ID_full = PIQ_with_ID[PIQ_with_ID[:, 1] != '']
PIQ = PIQ_with_ID_full[:, 1].astype(float)
bids_dir = 'E:\Work\MBS\EEG\Modelling\Intelligence\Cuban Map Project\Data\ds_bids_cbm_loris_24_11_21'
folders = glob.glob(bids_dir + '\sub-*', recursive=False)[:1]
subjects = [folder.removeprefix(bids_dir + '\sub-') for folder in folders]

fmin = (4, 4,  8,  8,  10, 13, 20, 30, 30, 4)
fmax = (8, 30, 13, 10, 13, 20, 30, 40, 45, 45)

conn_dir = 'Conn'
labels_dir = 'Labels'
source_ts_dir = 'Source time series'
clean_epochs_dir = 'Preprocessed data'

for subject, folder in zip(subjects, folders):
    # Skip already computed conns
    conn_files = glob.glob(f'{conn_dir}/*')
    conn_subjs = [file.removeprefix(f'{conn_dir}\\').split("_")[0] for file in conn_files]
    if subject in conn_subjs: continue
    #Also skip if subject has no WAIS data. 
    if subject not in PIQ_with_ID_full[:, 0]: continue
    # And also skip if subject has no EEG.
    if not any([x.endswith('eeg') for x in glob.glob(folder + '\\*')]): continue
    # The names are incorrect (lower case while folders are in upper case), so fix them.
    eeg_files = glob.glob(f'{folder}\eeg\sub-{subject.lower()}*')
    for eeg_file in eeg_files:
        os.rename(eeg_file, 'CBM'.join(eeg_file.rsplit('cbm', 1)))
    
    
    path = mne_bids.BIDSPath(root=bids_dir, subject=subject, datatype='eeg', task='protmap')
    params = {'preload':True}
    raw = mne_bids.read_raw_bids(path, extra_params=params, verbose=False)
    
    # And ALSO skip if record is shorter than expected (Yes, this f***ing dataset
    # has a lot of surprises). 
    if raw.times[-1] < 600: continue

    has_eog = False
    
    # Yes, this dataset has different set of electrodes. 
    if ('EOI' in raw.info['ch_names'] and 
        'EOD' in raw.info['ch_names'] and 
        'ECD' in raw.info['ch_names']):
        has_eog = True
        raw.set_channel_types({'EOI': 'eog',
                               'EOD': 'eog',
                               'ECG': 'ecg'})
    if ('EOG1' in raw.info['ch_names'] and 
        'EOG2' in raw.info['ch_names'] and 
        'DC1' in raw.info['ch_names']):
        has_eog = True
        raw.set_channel_types({'EOG1': 'eog',
                               'EOG2': 'eog',
                               'DC1': 'misc'})
    if ('Eiz' in raw.info['ch_names'] and 
        'Ede' in raw.info['ch_names'] and 
        'DC2' in raw.info['ch_names']):
        has_eog = True
        raw.set_channel_types({'Eiz': 'eog',
                               'Ede': 'eog',
                               'DC2': 'misc'})
    # Montage is in MNI space for some reason, so change the coordinates to a head space. 
    new_montage = raw.get_montage().copy()
    new_montage.add_mni_fiducials('fsaverage')
    trans = mne.channels.compute_native_head_t(new_montage)
    new_montage.apply_trans(trans)
    raw.set_montage(new_montage)
    
    raw_rest = raw.crop(tmin=0.0, tmax=600.0)
    raw_rest.set_eeg_reference('average', projection=True)
    raw_rest.apply_proj()
    raw_rest.filter(1, 45)
    raw_rest.resample(200.0)
    
    epochs = mne.make_fixed_length_epochs(raw_rest, duration=6., preload=True, overlap=1.)
    
    reject = autoreject.get_rejection_threshold(epochs, ch_types = ['eeg'], 
                                            random_state=42)
    n_components = raw_rest.info['nchan']-len(raw_rest.info['bads'])- 10
    ica = mne.preprocessing.ICA(n_components = n_components, method = 'infomax',
                                random_state=42)
    ica.fit(epochs)
    eog_channel = None if has_eog else 'Fp2' # Looks controversial but makes perfect sense - 
    # MNE automatically uses EOG channel as a channel to perform ocular artifact 
    # correction, so there's no need to pass any channel to find_bads_eog in this case.
    # Instead, if there's no such a channel, use Fp2 for this. 
    eog_inds, eog_scores = ica.find_bads_eog(epochs, ch_name = eog_channel, 
                                    measure = 'correlation', threshold = 0.4)
    muscle_inds, muscle_scores = ica.find_bads_muscle(epochs)
    if 'ecg' in epochs.get_channel_types():
        ecg_inds, ecg_scores = ica.find_bads_ecg(epochs)
        ica.exclude.extend(ecg_inds)
    ica.exclude.extend(eog_inds)
    ica.exclude.extend(muscle_inds)
    ica.apply(epochs)
    
    ar = autoreject.AutoReject(random_state=42)
    clean_epochs, rej_log = ar.fit_transform(epochs, return_log=True)
    
    rsc = autoreject.Ransac(verbose = False, n_resample=100, min_channels=0.2, min_corr=0.9, n_jobs = -1, random_state=42)
    clean_epochs = rsc.fit_transform(clean_epochs)
    
    # A dirty trick to prevent source rocenstruction from crash because of wrong 
    # reference projector due to removing bads in autorject.
    clean_epochs.set_eeg_reference('average', projection=True)
    clean_epochs.apply_proj()
    
    n_inter_chls = len(rsc.bad_chs_)
    
    labels, ts = source_utils.fsaverage_time_courses(clean_epochs, clean_epochs.info, method='eLORETA')
    
    conn = spectral_connectivity_epochs(ts, method='wpli', sfreq=raw_rest.info['sfreq'], 
                                        fmin=fmin, fmax=fmax, 
                                        faverage=True, n_jobs=-1)
    
    np.save(f'{conn_dir}/{subject}_ninters_{n_inter_chls}', conn)
    # This one (labels) GOTTA be one and the same for all subject, but for 
    # God's sake I save 'em all.
    np.save(f'{labels_dir}/{subject}_labels', labels) 
    np.save(f'{source_ts_dir}/{subject}_source_ts', ts) 
    clean_epochs.save(f'{clean_epochs_dir}/{subject}_clean_epo.fif', fmt='double')
    
    
    
    
    
    