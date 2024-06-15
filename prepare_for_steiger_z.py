# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 12:09:01 2024

@author: Admin
"""

# This script reads the leison results in order to apply steiger's Z test. 

import numpy as np
import glob
import os
from scipy import stats 
from sklearn.metrics import r2_score

CPM_order = 1
p_threshold = 0.001
method = 'imCoh' # PLV, wPLI or imCoh
sample = 'Chel' # Cuba, Chel, German
corr_mode = 'partial' # corr or partial
atlas = 'Destrieux' # Destrieux or DK
intel_test = 'LPS' # WST, LPS or RWT; applicable to German sample only
validation = 'LOO' # k-fold or LOO
if sample != 'German': intel_test = ''
prefix = 'neg'
freq = '10-13'


res_dir = res_dir = f'Results\\CPM results {validation}\\{method}\\{atlas} {sample}{intel_test}\\{corr_mode}\\{CPM_order} order\\{p_threshold} p_value\\'
label_dir = f'Labels\\{atlas}\\'
ROI_offset = 2 if atlas == 'Destrieux' else 1
ROI_labels = np.load(label_dir + 'CBM00001_labels.npy', allow_pickle=True)[:-ROI_offset]
intel_arr = np.load(f'{res_dir}\\intel_arr_{prefix}_{freq} Hz.npy')
pred = np.load(f'{res_dir}\\{prefix}_pred_{freq} Hz.npy')
whole_brain_r = stats.pearsonr(intel_arr, pred)
R2 = r2_score(intel_arr, pred)

lesion_results = glob.glob(res_dir + 'lesion_*')
# Lesion results are a little bit cursed organized, so here's the process of collecting
# the neccessary ones. Long story short, we pick those results that fit freq range and
# correlation sign.
specific_lesion_results = [result for result in lesion_results if os.path.exists(f'{result}\\all_{prefix}edges_{freq} Hz.npy')]
for result in specific_lesion_results:
    tr_indices = np.triu_indices(len(ROI_labels), k=1)
    idx = int(result.split('_')[-1].removeprefix('[').removesuffix(']'))
    edge_name = f'{ROI_labels[tr_indices[0][idx]].name} to {ROI_labels[tr_indices[1][idx]].name}'
    
    lesioned_intel_arr = np.load(f'{result}\\intel_arr_{prefix}_{freq} Hz.npy')
    lesioned_pred = np.load(f'{result}\\{prefix}_pred_{freq} Hz.npy')
    
    # The lengths of the whole-brain and lesioned arrays might not fit, so 
    # retain the least the intersection (could not do it with sets as values
    # might duplicate). 
    idcs = []
    l_idx = 0 # Index in the lesioned array.
    for i, el in enumerate(intel_arr):
        if el == lesioned_intel_arr[l_idx]:
            l_idx += 1
            idcs.append(i)
    # We suppose implicitly that intel_arr is the longest array and contains the whole 
    # lesioned_intel_arr and, probably, a bit more, so we just need to delete exceeding 
    # elements from it. This is most probably the case, but, technically, might be wrong,
    # so one more check there is. 
    if np.all(intel_arr[idcs] == lesioned_intel_arr):
        print('Arrays aligned, starting the calculations.')
    else:
        print('Arrays aligned but do not match, fix the bug :-(')
    lesioned_r = stats.pearsonr(lesioned_intel_arr, lesioned_pred)
    inter_pred_r = stats.pearsonr(lesioned_pred, pred[idcs])
    print(f'{len(lesioned_intel_arr)} elements presented.')
    print(f'For edge {edge_name}:')
    print(f'R^2 is {round(r2_score(lesioned_intel_arr, lesioned_pred), 3)}')
    print(f'Lesioned correlation: {round(lesioned_r[0], 3)}')
    print(f'Whole-brain correlation: {round(whole_brain_r[0], 3)}')
    print(f'Inter-prediction correlation: {round(inter_pred_r[0], 3)}')
    print('')
    