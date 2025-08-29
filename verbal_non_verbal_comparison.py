# -*- coding: utf-8 -*-
"""
Created on Fri May 30 14:54:47 2025

@author: Admin
"""

# This script compares the results of verbal and non-verbal intelligence prediction
# obtained on the 3rd sample. 

import scipy as sp
import numpy as np
from scipy.stats import pearsonr
import intel_utils
import matplotlib.pyplot as plt
import os
from openpyxl import load_workbook
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import glob

dataset = 'German'
test = 'RWT'

if dataset == 'Cuba':
    nonverbal_intel = intel_utils.get_WAIS_intel('PIQ')
    verbal_intel = intel_utils.get_WAIS_intel('VIQ')
elif dataset == 'German':
    nonverbal_intel = intel_utils.get_LPS_intel()
    verbal_intel = intel_utils.get_RWT_intel() 
# lps_intel = intel_utils.get_LPS_intel()
# rwt_intel = intel_utils.get_RWT_intel()

nonverbal_arr = []
vebral_arr = []
# lps_arr = []
# rwt_arr = []
subjects = set(nonverbal_intel.keys()).intersection(set(verbal_intel.keys()))
for subj in subjects:
    if not np.isnan(nonverbal_intel[subj]) and not np.isnan(verbal_intel[subj]):
        nonverbal_arr.append(nonverbal_intel[subj])
        vebral_arr.append(verbal_intel[subj])

print(pearsonr(nonverbal_arr, vebral_arr))
print('-----------------------------------')
plt.scatter(nonverbal_arr, vebral_arr)
plt.xlabel('Nonverbal values')
plt.ylabel('Verbal values')
plt.grid()
plt.show()

res_dir = 'H:\Work\Intelligence\Results\CPM results single LOO'
der_res_dir = 'H:\Work\Intelligence\Results\Derivative results'

Anton_results = f'{der_res_dir}\\final_results_corrected.xlsx'
wb = load_workbook(Anton_results)
ws = wb.active
for i, row in enumerate(ws.values):
    if i == 0: continue # Skip the header. 
    p_threshold = row[0]
    CPM_order = row[1]
    atlas = row[2]
    _dataset = row[3]
    method = row[4]
    corr_sign = row[5]
    freq = row[6]
    
    if _dataset == 'GermanLPS': _dataset = 'German' # Techichal debt.
    # Retain only the chosen results.
    if _dataset != dataset: continue
    
    rwt_res_dir = f'{res_dir}\\{method}\\{atlas} {dataset}{test}\\partial\\{CPM_order}\\{p_threshold}'
    intel_file = f'{rwt_res_dir}\\memory_{freq}.npy'
    pred_file = f'{rwt_res_dir}\\{corr_sign}_pred_{freq}.npy'
    all_res = glob.glob(f'{rwt_res_dir}\\{corr_sign}_pred_*')
    # N is a number of results obtained with the specific set of parameters; used to calculated corrected p-value.
    N = len(all_res)
    
    # If there's not a result with the same parameters in the verbal intelligence case, then skip.
    if not os.path.exists(pred_file): continue
    rwt_intel = np.load(intel_file)
    rwt_pred = np.load(pred_file)
    r, p = pearsonr(rwt_intel, rwt_pred)
    mae = mean_absolute_error(rwt_intel, rwt_pred)
    rmse = root_mean_squared_error(rwt_intel, rwt_pred)
    
    # Look for the valuable edges and list them.
    edges = np.load(f'{res_dir}\\{method}\\{atlas} {dataset}{test}\\partial\\{CPM_order}\\' + \
                    f'{p_threshold}\\{corr_sign}edges_{freq}.npy')
    label_dir = f'Labels\\{atlas}\\'
    ROI_offset = 2 if atlas == 'Destrieux' else 1
    ROI_labels = np.load(label_dir + 'CBM00001_labels.npy', allow_pickle=True)[:-ROI_offset]
    ROI_names = [label.name for label in ROI_labels]
    matrix = sp.spatial.distance.squareform(edges)
    idcs = np.argwhere(matrix)
    tril_idcs = np.array(np.tril_indices(matrix.shape[0])).T
    # I guess, not the most obvious way to remove duplicating edges, but whatever, it works.
    idcs = set([tuple(x) for x in idcs]) & set([tuple(x) for x in tril_idcs])
    verbal_edge_names = set([f'{ROI_names[idx[0]]} to {ROI_names[idx[1]]}'for idx in idcs])
    nonverbal_edges_file = f'{der_res_dir}\\Edges\\{method}_{atlas}_{dataset}{test}_{CPM_order}_{p_threshold}_{corr_sign}_{freq}_edges.csv'
    nonverbal_edge_names = set([edge[0] for edge in intel_utils.read_csv(nonverbal_edges_file)[1:]])
    edge_intersection = set.intersection(verbal_edge_names, nonverbal_edge_names)
    print(f'Frequency: {freq}, {p_threshold}, {CPM_order}, atlas: {atlas}, FC: {method},\n' + \
          f'corr sign: {corr_sign}, r: {round(r, 5)}, corrected p: {round(p * N, 6)}, MAE: {round(mae, 2)}, RMSE: {round(rmse, 2)}')
    print('^^^^^^^^^^^^^^^^^^^^^^')
    print('\n'.join(edge_intersection))
    print('^^^^^^^^^^^^^^^^^^^^^^')
    

    