# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:27:48 2024

@author: Admin
"""

# This file draws bar plots for lesioned results. 

import numpy as np
import glob
import os
from scipy import stats 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

CPM_order = 1
p_threshold = 0.001
method = 'imCoh' # PLV, wPLI or imCoh
sample = 'Chel' # Cuba, Chel, German
corr_mode = 'partial' # corr or partial
atlas = 'Destrieux' # Destrieux or DK
intel_test = 'LPS' # WST, LPS or RWT; applicable to German sample only
validation = 'LOO' # k-fold or LOO
if sample != 'German': intel_test = ''
prefix = 'pos'
freq = '10-13'


res_dir = res_dir = f'Results\\CPM results {validation}\\{method}\\{atlas} {sample}{intel_test}\\{corr_mode}\\{CPM_order} order\\{p_threshold} p_value\\'
img_dir = f'Results\\CPM plots {validation}\\{method}\\{atlas} {sample}{intel_test}\\{corr_mode}\\{CPM_order} order\\{p_threshold} p_value\\'

label_dir = f'Labels\\{atlas}\\'
ROI_offset = 2 if atlas == 'Destrieux' else 1
ROI_labels = np.load(label_dir + 'CBM00001_labels.npy', allow_pickle=True)[:-ROI_offset]
intel_arr = np.load(f'{res_dir}\\intel_arr_{prefix}_{freq} Hz.npy')
pred = np.load(f'{res_dir}\\{prefix}_pred_{freq} Hz.npy')
whole_brain_r = stats.pearsonr(intel_arr, pred)
R2 = stats.pearsonr(intel_arr, pred)[0]

# 1st sample, imCoh, 10-13 Hz, positively correlated edges
steiger_p_values = [0.016, 0.003, 0.014, 0.003, 0.002, 0.060, 0.251, 0.058]
# 2nd sample, imCoh, 10-13 Hz, positively correlated edges;
# steiger_p_values = [0.010, 0.001, 0.205, 0.047, 0.005]
# 3rd sample, wPLI, 4-8 Hz, negatively correlated edges
# steiger_p_values = [0.003, 0.33]
# 1st sample, imCoh, 10-13 Hz, negatively correlated edges
# steiger_p_values = [0.314, 0.391, 0.057, 0.010]
# 1st sample, imCoh, 10-13 Hz, positively correlated edges
# steiger_p_values = [0.212, 0.097, 0.011, 0.471]
# 2nd sample, PLV, 8-10 Hz, positively correlated edges
# steiger_p_values = [0.075, 0.100, 0.693, 0.017]
# 2nd sample, PLV, 10-13 Hz, positively correlated edges
# steiger_p_values = [0.014, 0.0009]
# 2nd sample, PLV, 13-20 Hz, positively correlated edges
# steiger_p_values = [0.008, 0.122, 0.016]
# 2nd sample, PLV, 20-30 Hz, positively correlated edges
# steiger_p_values = [0.0009, 0.019, 0.178, 0.003]
# 2nd sample, imCoh, 8-10 Hz, positively correlated edges
# steiger_p_values = [0.003, 0.010]

lesion_results = glob.glob(res_dir + 'lesion_*')
# Lesion results are a little bit cursed organized, so here's the process of collecting
# the neccessary ones. Long story short, we pick those results that fit freq range and
# correlation sign.
specific_lesion_results = [result for result in lesion_results if os.path.exists(f'{result}\\all_{prefix}edges_{freq} Hz.npy')]
r2s = [R2] # Not r-squared values anymore, but I don't bother to change the name. 
edge_names = ['Whole brain model']
for result in specific_lesion_results:
    tr_indices = np.triu_indices(len(ROI_labels), k=1)
    idx = int(result.split('_')[-1].removeprefix('[').removesuffix(']'))
    edge_name = f'{ROI_labels[tr_indices[0][idx]].name} to \n {ROI_labels[tr_indices[1][idx]].name}'
    edge_names.append(edge_name)
    
    lesioned_intel_arr = np.load(f'{result}\\intel_arr_{prefix}_{freq} Hz.npy')
    lesioned_pred = np.load(f'{result}\\{prefix}_pred_{freq} Hz.npy')
    r2s.append(stats.pearsonr(lesioned_intel_arr, lesioned_pred)[0])

plot_data =  pd.DataFrame(np.array([r2s, edge_names], dtype=object).T, columns = ['Correlation coefficient', 'Removed valuable edges'])
fig, ax = plt.subplots(figsize = (8, 4))
ax = sns.barplot(plot_data, x = 'Removed valuable edges', y = 'Correlation coefficient', width = 0.4, dodge = False, saturation=0.4, ax = ax)
# ax = sns.barplot(plot_data, x = 'Removed valuable edges', y = 'Correlation coefficient', width = 0.8, dodge = False, saturation=0.4)
# ax.set_xlim([0, 10])
# ax.set_aspect(1.0)
ax.set_xlabel(ax.get_xlabel(), fontsize = 16)
ax.set_ylabel(ax.get_ylabel(), fontsize = 16)
ax.tick_params(axis = 'x', labelsize = 16)
ax.tick_params(axis = 'y', labelsize = 16)
ax.tick_params(axis='x', rotation=90)
labels = [str(round(r2, 2)) for r2 in r2s]
for i, label in enumerate(labels):
    if i == 0: continue
    p = steiger_p_values[i - 1]
    text = ''
    if p < 0.05:
        if (p <= 0.05) and (p > 0.01): 
            text = '*'
        elif (p <= 0.01) and (p > 0.001):
            text = '**'
        elif p <= 0.001:
            text = '***'
    labels[i] += text
ax.bar_label(ax.containers[0], fontsize=16, labels = labels)

h = 1.2
ax.set_ylim(top = (h) * max(r2s))
ax.figure.savefig(img_dir + f'[freq] Hz lesion_aggr_{prefix}_bar', bbox_inches='tight')