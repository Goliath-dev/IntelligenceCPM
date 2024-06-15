# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 21:20:52 2023

@author: Admin
"""

from mne import compute_covariance, setup_source_space, make_forward_solution 
from mne import read_labels_from_annot, extract_label_time_course
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.datasets import fetch_fsaverage
import os.path as op

def fsaverage_time_courses(epochs, info, method='dSPM', snr=3, epoch_count=None, 
                           parc = 'aparc', verbose=False):
    """
    

    Parameters
    ----------
    epochs : Epochs
        The Epochs object to compute the sources from. 
    info : Info
        The Info object. Not actually needed as an argument, just pass epochs.Info here.
    method : string, optional
        The method of source reconstruction. The available options are 'MNE',
        'dSPM', 'eLORETA' and 'sLORETA'. The default is 'dSPM'.
    snr : int, optional
        Signal to noise ratio. The default is 3.
    epoch_count : int, optional
        The count of epochs to compute source from. If None, all epochs are applied,
        instead the first epoch_count epochs are used. Mainly for technical purposes
        only, pass a small number here if some fast check is needed. The default is None.
    parc : string, optional
        A parcellation to divide a brain into. Pass 'aparc' for the Dessikan-Killiany
        atlas, 'aparc.a2009s' for the Destrieux one. The default is 'aparc'.
    verbose : bool, optional
        The verbosity status. The default is False.

    Returns
    -------
    labels : A list of Label 
        A list of Label objects containing information about brain regions.
    time_courses : ndarray
        An array of shape n_labels x n_time_points.

    """
    noise_cov = compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'], verbose=verbose)
    fs_dir = fetch_fsaverage(verbose=verbose)
    subjects_dir = op.dirname(fs_dir)
    subject = 'fsaverage'
    trans = 'fsaverage'
    src = setup_source_space(subject=subject, subjects_dir=subjects_dir, n_jobs=-1)
    bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    fwd = make_forward_solution(info, trans=trans, src=src, 
                                    eeg=True, bem=bem, mindist=5.0, n_jobs=-1)
    inverse = make_inverse_operator(info, fwd, noise_cov, loose=0.2, depth=0.8)
    lambda2 = 1. / snr ** 2
    stc = None
    if epoch_count == None:
        stc = apply_inverse_epochs(epochs, inverse, lambda2, method=method, verbose=verbose)
    else:
        stc = apply_inverse_epochs(epochs[:epoch_count], inverse, lambda2, method=method, verbose=verbose)
    labels = read_labels_from_annot(subject=subject, subjects_dir=subjects_dir, parc = parc)
    offset = 1 if parc == 'aparc' else 2
    time_courses = extract_label_time_course(stc, labels[:-offset], src)
    return labels, time_courses