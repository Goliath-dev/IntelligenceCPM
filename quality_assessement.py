# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 19:22:36 2025

@author: Admin
"""

# This script is a translation of the original MATLAB functions of PaLOSi. 

import numpy as np
from scipy.signal.windows import dpss
from scipy.fft import fft
import os
import matplotlib.pyplot as plt
import mne  # For topomaps (if needed)
import glob
import seaborn as sns
import pandas as pd
import csv

class EEGDecorator:
    def __init__(self, mne_epochs):
        _data = mne_epochs.get_data(copy=True)
        _data = np.swapaxes(_data, 0, 2)
        _data = np.swapaxes(_data, 0, 1)
        self._data = _data
        self._srate = mne_epochs.info['sfreq']
    
    def __getitem__(self, key):
        if key == 'data':
            return self._data
        elif key == 'srate':
            return self._srate
        
    # @property
    # def data(self):
    #     return self._data
    
    # @property
    # def srate(self):
    #     return self._srate
    
    
def CPCstepwise1(S, n, pmax, lmax):
    """
    Python translation of MATLAB's CPCstepwise1.m.
    Computes Common Principal Components (CPCs) for covariance matrices.

    Args:
        S (np.ndarray): Covariance matrices (shape p x p x k, where p=channels, k=frequencies)
        n (np.ndarray): Number of samples for each covariance matrix (shape k,)
        pmax (int): Number of CPCs to estimate
        lmax (int): Maximum iterations for each CPC

    Returns:
        Lambda (np.ndarray): Eigenvalues (pmax x k)
        Q (np.ndarray): Eigenvectors (p x pmax)
    """
    n = np.array(n).flatten()
    p, _, k = S.shape
    n = n.reshape(1, 1, k)
    nt = np.sum(n)
    
    # Step 1: Compute pooled covariance
    Spooled = np.sum(S * (n / nt), axis=2)
    
    # Step 2: Initial eigenvectors from pooled covariance
    eigvals, qtilde = np.linalg.eig(Spooled)
    # Sort eigenvectors by eigenvalues (descending)
    idx = np.argsort(eigvals)[::-1]
    qtilde = qtilde[:, idx]
    
    Pi = np.eye(p)
    Lambda = np.zeros((p, k), dtype='complex')
    Q = np.zeros((p, p), dtype='complex')
    mu = np.zeros((1, 1, k), dtype='complex')
    
    for j in range(pmax):
        x = qtilde[:, j]
        x = x / np.sqrt(x.T @ x)
        x = Pi @ x
        
        # print(x.T.dtype)
        # Update mu
        for i in range(k):
            mu[0, 0, i] = x.T @ S[:, :, i] @ x
        
        # Iterative refinement
        for _ in range(lmax):
            Spooled = np.sum(S * (n / mu), axis=2)
            y = Pi @ Spooled @ x
            x = y / np.sqrt(y.T @ y)
            
            for i in range(k):
                mu[0, 0, i] = x.T @ S[:, :, i] @ x
        
        Q[:, j] = x
        Pi = Pi - np.outer(x, x)
    
    # Compute final eigenvalues
    for i in range(k):
        # print((Q.T @ S[:, :, i]).shape)
        # print(Lambda[:, i].shape)
        Lambda[:, i] = np.diag(Q.T @ S[:, :, i] @ Q)
    
    # Return only the requested components
    return Lambda[:pmax, :], Q[:, :pmax]

def xspt(data, nw, fs, fmax=None, fmin=None):
    """
    Python translation of MATLAB's xspt.m.
    Computes multitaper cross-spectral matrix for multivariate EEG data.

    Args:
        data (np.ndarray): EEG data (shape: n_channels × n_times × n_segments).
        nw (float): Time-bandwidth product (e.g., 2, 3, etc.).
        fs (int): Sampling rate (Hz).
        fmax (float, optional): Max frequency to retain. Defaults to Nyquist.
        fmin (float, optional): Min frequency to retain. Defaults to 1 Hz.

    Returns:
        Pxy (np.ndarray): Cross-spectral matrix (n_channels × n_channels × n_freqs).
        f (np.ndarray): Frequency bins (n_freqs,).
        nss (int): Effective number of segments (ns * (2*nw - 1)).
    """
    # Input checks and reshape if single segment
    if data.ndim == 2:
        data = data[:, :, np.newaxis]  # Add segment axis
    nc, nt, ns = data.shape

    # Default FFT parameters (match MATLAB's nfft = 2.56*fs)
    nfft = int(2.56 * fs)
    if ns == 1:
        ns = nt // nfft
        data = data[:, :ns * nfft].reshape((nc, nfft, ns))

    # Frequency bins
    f = np.arange(1, nfft + 1) * (fs / nfft)
    fftidx = np.arange(nfft)  # Default: all frequencies

    # Frequency masking (if fmin/fmax provided)
    if fmax is not None or fmin is not None:
        fmin = 1 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        fftidx = np.where((f >= fmin) & (f <= fmax))[0]
        f = f[fftidx]
    nf = len(fftidx)

    # Generate Slepian sequences (DPSS tapers)
    n_tapers = int(2 * nw) - 1  # Number of tapers
    # E = dpss(nfft, nw, Kmax=n_tapers)  # (n_tapers × nfft)
    E = dpss(nt, nw, Kmax=n_tapers)  # (n_tapers × nfft)
    E = E.T  # (nfft × n_tapers)
    E = np.delete(E, -1, axis = -1)
    E = np.tile(E[np.newaxis, :, :], (nc, 1, 1))  # (nc × nfft × n_tapers)

    # Compute cross-spectral matrix
    Pxy = np.zeros((nc, nc, nf), dtype=np.complex128)
    for i in range(ns):
        dati = data[:, :, i]  # (nc × nfft)
        dati_tapered = np.tile(dati[:, :, np.newaxis], (1, 1, E.shape[2])) * E  # (nc × nfft × n_tapers)
        
        # FFT and select frequencies
        fc_i = fft(dati_tapered, axis=1)  # (nc × nfft × n_tapers)
        # print(fc_i.shape)
        fc_i = fc_i[:, fftidx, :]  # (nc × nf × n_tapers)
        # print(fftidx)
        # print(fc_i.shape)
        
        # Covariance at each frequency (non-conjugate transpose)
        for j in range(nf):
            fc_j = fc_i[:, j, :]  # (nc × n_tapers)
            # print(fc_j.shape)
            Pxy[:, :, j] += np.cov(fc_j, rowvar=True, bias=True)  # (nc × nc)

    # Normalize by effective number of segments
    nss = ns * (2 * nw - 1)
    Pxy = Pxy / nss

    return Pxy, f, nss


def qcspectra(EEG, nw, fs, fmax=30, fmin=0.99):
    """
    Python translation of MATLAB's qcspectra.m.
    Computes cross-spectral quality metrics for EEG data.

    Args:
        EEG (np.ndarray): EEG data (shape: n_channels × n_times [× n_epochs]).
        nw (float): Time-halfbandwidth product (for multitaper).
        fs (int): Sampling rate (Hz).
        fmax (float): Max frequency for analysis (default: 30 Hz).
        fmin (float): Min frequency for analysis (default: 0.99 Hz).
        chanlocs (list/dict): Channel locations (for plotting).
        svpath (str): Save path for plots (optional).

    Returns:
        pro (float): PaLOSi index (proportion of variance explained by 1st CPC).
        rkr (list): Quality metrics [rank(EEG), rank(PSD), rank(cross-spectrum at 10Hz), mean correlation].
        f (np.ndarray): Frequencies.
        ssd (float): Total power (sum of PSD).
    """
    # print("Into qcspectra")
    # Check input dimensions
    if EEG.shape[0] < 2:
        return 0, [0, 0, 0, 0], np.array([]), 0

    # Compute cross-spectrum (placeholder for xspt)
    S, f, nss = xspt(EEG, nw, fs, fmax, fmin)
    # print('xspt passed')
    fbd = [fmin, fmax]
    nf = S.shape[2]
    # print(S.shape)
    n = nss * np.ones(nf)

    # CPC decomposition (placeholder for CPCstepwise1)
    lmd, Q = CPCstepwise1(S, n, pmax=10, lmax=50)
    lmd = np.real(lmd.T)  # Shape: freq × CPs

    # Power spectral density (PSD)
    psd = np.abs(np.diagonal(S, axis1=0, axis2=1).T)  # Sum over channels
    ssd = np.sum(psd)  # Total power

    # PaLOSi metrics
    profd = lmd[:, 0] / ssd  # Frequency-dependent PaLOSi (1st CPC)
    procd = np.sum(lmd, axis=0) / ssd  # Component-dependent PaLOSi
    pro = np.sum(profd)  # Total PaLOSi

    # Plot results (if chanlocs and svpath provided)
    # if chanlocs is not None and svpath is not None:
    #     svfd, nm = os.path.split(svpath)
        # plt_paramap_py(lmd, Q, procd, fbd, pmax=10, svfd=svfd, nm=nm, chanlocs=chanlocs, hr=0.5)

    # Quality metrics
    rou = np.triu(np.corrcoef(np.log10(psd)), 1)
    mr = np.sum(rou) / (len(f) * (len(f) - 1) / 2)
    fid10 = np.argmin(np.abs(f - 10))
    rkr = [
        np.linalg.matrix_rank(EEG),
        np.linalg.matrix_rank(psd),
        np.linalg.matrix_rank(S[:, :, fid10]),
        mr
    ]

    return pro, rkr, f, ssd


def ck_palos(sample, srate=500, sbjset=None):
    """
    Python translation of MATLAB's ck_palos.m.
    Computes PaLOSi (Pattern of Local Oscillatory Stability) from EEG data.

    Args:
        eegpath (str): Path to EEG data (EEGLAB struct or Automagic results).
        srate (int, optional): Sampling rate. Defaults to 500.
        sbjset (list, optional): Indices of subjects to process. Defaults to all.

    Returns:
        pro (np.ndarray): PaLOS index for each subject.
        sbjnm (list): Subject names.
        spectra (np.ndarray): Power spectra.
        freqs (np.ndarray): Frequencies.
        speccomp (np.ndarray): Spectral components.
        contrib (np.ndarray): Contributions.
        specstd (np.ndarray): Spectral standard deviations.
    """
    # Create save directory
    # svfd = f"{eegpath}_ckpls"
    # os.makedirs(svfd, exist_ok=True)

    # Find EEG files (Automagic or EEGLAB)
    # eegset = glob(f"{eegpath}/**/all*.mat", recursive=True)  # Automagic
    # if not eegset:
    #     eegset = glob(f"{eegpath}/*.mat")  # EEGLAB
    sample_dict = {'Chel': ('E:\Work\MBS\EEG\Modelling\Intelligence\Preprocessed data\Destrieux', 'chcon'),
                   'Cuba': ('E:\Work\MBS\EEG\Modelling\Intelligence\Cuban Map Project\Preprocessed data', 'CBM'),
                   'German': ('D:\EEG_data\German dataset\My derivatives\Epochs', 'sub')}
    eegset = glob.glob(f'{sample_dict[sample][0]}\\{sample_dict[sample][1]}*.fif')
    # print(eegset)

    # Partial check (subset of subjects)
    if sbjset is None:
        sbjset = range(len(eegset))
    nsbj = len(sbjset)

    # Initialize outputs
    # pro = np.zeros(nsbj)
    pro = []
    sbjnm = [""] * nsbj
    spectra, freqs, speccomp, contrib, specstd = [], [], [], [], []

    for i, j in enumerate(sbjset):
        # Load EEG data
        print(f">>>>-------------------processing sbj: {i+1}/{nsbj}")
        eegfile = eegset[j]
        nmn = os.path.basename(eegfile).replace(".mat", "")
        print(f"Sbjname:     {nmn}")
        sbjnm[i] = nmn

        # Load .mat file (EEGLAB/Automagic)
        # eeg_data = scipy.io.loadmat(eegfile)
        # if "EEGFinal" in eeg_data:  # Automagic
        #     EEG = eeg_data["EEGFinal"]
        # else:  # EEGLAB
        #     EEG = eeg_data["EEG"]
        eeg_data = mne.read_epochs(eegfile)
        EEG = EEGDecorator(eeg_data)

        # Skip bad channels
        # idx = (EEG["data"][:, 0] != 0) & (~np.isnan(EEG["data"][:, 0]))
        # data = EEG["data"][idx, :]
        data = EEG["data"]
        # # chanlocs = EEG["chanlocs"][idx]
        # if data.shape[0] < 2:
        #     print(f"Skip:     {eegfile}")
        #     continue

        # Update sampling rate if needed
        # if "srate" in EEG and srate != EEG["srate"]:
        #     srate = EEG["srate"]
        srate = EEG['srate']

        # Compute PaLOSi (qcspectra equivalent)
        # pro[i] = qcspectra(data, nw=3, fs=srate, fmax=30, fmin=0.99, 
        #                      chanlocs=None, svpath=f"{svfd}/{nmn}")
        pro.append(qcspectra(data, nw=3, fs=srate, fmax=30, fmin=0.99))

        # Plot EEG time series (eegplot_w equivalent)
        # fig = plt.figure(figsize=(12, 6))
        # plt.plot(data.T)
        # plt.title(nmn)
        # plt.savefig(f"{svfd}/wv_{nmn}.png")
        # plt.close()

        # Plot spectral topography (spectopo equivalent)
        # fig = plt.figure(figsize=(8, 6.5))
        # spectra_i, freqs_i = mne.time_frequency.psd_array_welch(
        #     data, sfreq=srate, fmin=0.25, fmax=30.5, n_fft=int(2.56 * srate))
        # plt.plot(freqs_i, 10 * np.log10(spectra_i.mean(axis=0)))  # dB
        # plt.xlabel("Frequency (Hz)")
        # plt.ylabel("Power (dB)")
        # plt.title(nmn)
        # plt.savefig(f"{svfd}/spt_{nmn}.svg")
        # plt.close()

        # Store spectra (simplified; adjust as needed)
        # spectra.append(spectra_i)
        # freqs.append(freqs_i)

    # Save results
    # np.savez(f"{svfd}/z_pro.npz", pro=pro, sbjnm=sbjnm)
    return pro, sbjnm, spectra, freqs, speccomp, contrib, specstd



# Compute the PaLOSi metric for all datasets. 
samples = ['Chel', 'Cuba', 'German']
sample_dict = {'Chel': 'First\ndataset', 'Cuba': 'Second\ndataset', 'German': 'Third\ndataset'}
total_palosi = []
total_samples = []
palosi_by_sample = {'Chel': [], 'Cuba':[], 'German': []}
for sample in samples:
    res = ck_palos(sample)
    palosi_arr = [res[0][i][0] for i in range(len(res[0]))]
    palosi_by_sample[sample] = palosi_arr
    total_palosi.extend(palosi_arr)
    total_samples.extend([sample_dict[sample]] * len(palosi_arr))
    plt.hist(palosi_arr)
    plt.title(f'PaLOSi on {sample}, median = {np.median(palosi_arr)}')
    plt.show()

# Draw a figure.
img_dir = 'E:\Work\MBS\EEG\Modelling\Intelligence\Results\Imgs for article'
data = pd.DataFrame(data=np.array([total_palosi, total_samples], dtype='object').T, columns=['PaLOSi', 'Samples'])
# fig, ax = plt.subplots(figsize=(40, 20))
ax = sns.boxplot(data, x='Samples', y='PaLOSi')    
ax.plot([-1, 3], [0.3, 0.3], 'k--')
ax.plot([-1, 3], [0.6, 0.6], 'k--')
ax.set_yticks([0, 0.3, 0.6, 1])
ax.set(xlabel=None)
ax.figure.savefig(f'{img_dir}\\quality.png', dpi=600)

# Save results in files. Could've done it in a cycle above, but decided to separate for readability. 
der_res_dir = 'H:\Work\Intelligence\Results\Derivative results'
for sample in samples:
    data = palosi_by_sample[sample]
    with open(f'{der_res_dir}\\PaLOSi\\{sample}.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter = ',')
        writer.writerow([f'{sample}'])
        for el in data:
            writer.writerow([el])
        