# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 21:34:46 2025

@author: Admin
"""


from philistine.mne import savgol_iaf
import glob
import mne
import numpy as np
import matplotlib.pyplot as plt
import intel_utils
import mne_bids

def read_raw_data(sample, subj):
    if sample == 'Chel':
        # Change the format of ID from chconN to chcon_f_N. God knows why I changed it, but hey, revert back. 
        raw_subj_name = f'{subj[:5]}_f_{subj[5:]}' 
        file = f'D:\\EEG_data\\Connectivity\\CLEAR_DATA\\EEG\\{raw_subj_name}.vhdr'
        raw = mne.io.read_raw_brainvision(file, preload=True, verbose=False)
        return raw
    elif sample == 'Cuba':
        bids_dir = 'E:\\Work\\MBS\\EEG\\Modelling\\Intelligence\\Cuban Map Project\\Data\\ds_bids_cbm_loris_24_11_21'
        path = mne_bids.BIDSPath(root=bids_dir, subject=subj, datatype='eeg', task='protmap')
        params = {'preload':True}
        raw = mne_bids.read_raw_bids(path, extra_params=params, verbose=False)
        return raw
    elif sample == 'German':
        file = f'D:\\EEG_data\\German dataset\\MPI-Leipzig_Mind-Brain-Body-LEMON\\EEG_MPILMBB_LEMON\\EEG_Raw_BIDS_ID\\{subj}\\RSEEG\\{subj}.vhdr'
        raw = mne.io.read_raw_brainvision(file, preload=True, verbose=True)
        return raw
        
# The following code repeats the piece of code from CPM_base.py to collect 
# the participants' IDs identical to those used there (essentially to filter out
# participants without intel values and fit age requirements). Most of the parameters
# are irrelevant in this case, so all but sample are just chosen randomly as they 
# make no difference whatsoever. 

# Parameters. 
CPM_order = 2
p_threshold = 0.001
method = 'imCoh' # PLV, wPLI or imCoh
sample = 'Cuba' # Cuba, Chel, German
corr_mode = 'partial' # corr or partial
atlas = 'Destrieux' # Destrieux or DK
intel_test = 'LPS' # WST, LPS or RWT; applicable to German sample only
validation = 'LOO' # k-fold or LOO
if sample != 'German': intel_test = ''
intel_method_dict = {'Chel': intel_utils.get_Raven_intel,
                     'Cuba': intel_utils.get_WAIS_intel,
                     'GermanLPS': intel_utils.get_LPS_intel,
                     'GermanWST': intel_utils.get_WST_intel,
                     'GermanRWT': intel_utils.get_RWT_intel}
age_method_dict = {'Chel': intel_utils.get_Raven_age,
                   'Cuba': intel_utils.get_WAIS_age,
                   'German': intel_utils.get_German_age}
# raw_read_method_dict = {'Chel': mne.io.read_raw_brainvision,
#                    'Cuba': mne_bids.read_raw_bids,
#                    'German': mne.io.read_raw_brainvision}

intel_method = intel_method_dict[sample + intel_test]
age_method = age_method_dict[sample]

res_dir = f'Results\\CPM results {validation}\\{method}\\{atlas} {sample}{intel_test}\\{corr_mode}\\{CPM_order} order\\{p_threshold} p_value\\'

behav = intel_method()
age = age_method()

conn_dir = f'Matrices\\{method}\\{atlas}\\{sample}\\'
files = glob.glob(conn_dir + '*.npy')

label_dir = f'Labels\\{atlas}\\'
subj_arr = [] 

for i, file in enumerate(files):
    subj = file.split('\\')[-1].split('.')[0]
    
    if np.isnan(age[subj]): 
        print(f'Subject {subj} does not have age data and is skipped.')
        continue
    if age[subj] > 35:
        print(f'Subject {subj} does not fit age requirement and is skipped.')
        continue
    if np.isnan(behav[subj]):
        print(f'Subject {subj} is missing intel data or the data is considered inappropriate.')
        continue
    subj_arr.append(subj)
    # print(sex[subj])



alpha_peaks = []
for subj in subj_arr:
    raw = read_raw_data(sample, subj)
    alpha_peaks.append(savgol_iaf(raw, fmin = 8, fmax = 13, pink_max_r2 = 0.97))

alphas = [peak.PeakAlphaFrequency for peak in alpha_peaks]
filtered = np.array(list(filter(lambda alpha: alpha is not None, alphas)))
plt.show()
plt.hist(filtered)
plt.title(f'Sample: {sample}, median: {np.median(filtered)}, ratio = {len(filtered)} / {len(alphas)}')