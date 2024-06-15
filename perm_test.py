# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 12:28:26 2023

@author: Admin
"""

import numpy as np
import random
from sklearn.metrics import r2_score

# This script performs a permutation test with given data.

CPM_order = 1
p_threshold = 0.001
method = 'imCoh' # PLV, wPLI or imCoh
sample = 'German' # Cuba, Chel or German
corr_mode = 'partial' # corr or partial
atlas = 'DK' # Destrieux or DK
freq = '20-30' # 4-8, 8-13, 8-10, 10-13, 13-20, 20-30 or 30-45
prefix = 'pos' # posedges or negedges
intel_test = 'LPS' # WST, LPS or RWT; applicable to German sample only
validation = 'LOO' # k-fold or LOO
if sample != 'German': intel_test = ''

n_perms = 10000

res_dir = f'Results\\CPM results {validation}\\{method}\\{atlas} {sample}{intel_test}\\{corr_mode}' +\
f'\\{CPM_order} order\\{p_threshold} p_value\\'
pred_file = f'{res_dir}{prefix}_pred_{freq} Hz.npy'
intel_file = f'{res_dir}intel_arr_{prefix}_{freq} Hz.npy'

pred = np.load(pred_file)
intel = np.load(intel_file)
R2_arr = np.zeros(n_perms)

for i in range(n_perms):
    N = len(pred)
    rand_idcs = np.arange(0, N)
    random.shuffle(rand_idcs)
    permuted_pred = pred[rand_idcs]
    R2_arr[i] = r2_score(intel, permuted_pred)
    
true_R2 = r2_score(intel, pred)
p = len(R2_arr[R2_arr > true_R2]) / n_perms
print(p)
    