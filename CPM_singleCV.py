# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 20:07:22 2025

@author: Admin
"""

import CPM_fixed
import intel_utils
import glob
import numpy as np
from sklearn.metrics import r2_score
import time
import os
from itertools import product

def main():    
    start = time.time()
    
    # Parameters. 
    CPM_orders = [1, 2, 3]
    p_thresholds = [0.01, 0.001]
    methods = ['imCoh', 'PLV', 'wPLI'] # PLV, wPLI or imCoh
    # samples = ['Chel', 'Cuba', 'German'] # Cuba, Chel, German
    samples = ['Cuba']
    corr_mode = 'partial' # corr or partial
    atlases = ['Destrieux', 'DK'] # Destrieux or DK
    # CPM_orders = [1]
    # p_thresholds = [0.01]
    # methods = ['PLV'] # PLV, wPLI or imCoh
    # samples = ['Chel'] # Cuba, Chel, German
    # corr_mode = 'partial' # corr or partial
    # atlases = ['Destrieux'] # Destrieux or DK    
    for CPM_order, p_threshold, method, sample, atlas in product(CPM_orders, p_thresholds, methods, samples, atlases):
        intel_test = 'VIQ' # WST, LPS or RWT for German; VIQ, FSIQ or PIQ (also nothing for this case) for Cuba; nothing for Chel
        validation = 'LOO' # k-fold or LOO
        # if sample != 'German': intel_test = ''
        intel_method_dict = {'Chel': intel_utils.get_Raven_intel,
                             'Cuba': lambda: intel_utils.get_WAIS_intel('PIQ'),
                             'CubaVIQ': lambda: intel_utils.get_WAIS_intel('VIQ'),
                             'GermanLPS': intel_utils.get_LPS_intel,
                             'GermanWST': intel_utils.get_WST_intel,
                             'GermanRWT': intel_utils.get_RWT_intel}
        age_method_dict = {'Chel': intel_utils.get_Raven_age,
                           'Cuba': intel_utils.get_WAIS_age,
                           'German': intel_utils.get_German_age}
        sex_method_dict = {'Chel': intel_utils.get_Raven_sex,
                           'Cuba': intel_utils.get_WAIS_sex,
                           'German': intel_utils.get_German_sex}
        
        
        intel_method = intel_method_dict[sample + intel_test]
        age_method = age_method_dict[sample]
        sex_method = sex_method_dict[sample]
        
        res_dir = f'H:\\Work\\Intelligence\\Results\\CPM results single {validation}\\{method}\\{atlas} {sample}{intel_test}\\{corr_mode}\\{CPM_order} order\\{p_threshold} p_value\\'
        
        behav = intel_method()
        age = age_method()
        sex = sex_method()
        
        conn_dir = f'Matrices\\{method}\\{atlas}\\{sample}\\'
        files = glob.glob(conn_dir + '*.npy')
        
        matrices_list = []
        complete_intel_arr = []
        complete_age_arr = []
        sex_arr = [] # Purely for descriptive purposes. 
        subj_arr = [] # Purely for descriptive purposes. 
        
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
            conn_data = np.load(file)
            low_tri_conn = np.array([row[row != 0] for row in conn_data])
            matrices_list.append(low_tri_conn)
            
            complete_intel_arr.append(behav[subj])
            complete_age_arr.append(age[subj])
            sex_arr.append(sex[subj])
            subj_arr.append(subj)
            # print(sex[subj])
        
        matrices = np.vstack(matrices_list)
        complete_intel_arr = np.array(complete_intel_arr)
        complete_age_arr = np.array(complete_age_arr)
        
        
        
        # Validation and plotting.
        fmin = (4, 4,  8,  8,  10, 13, 20, 30, 30, 4)
        fmax = (8, 30, 13, 10, 13, 20, 30, 40, 45, 45)
        freq_idcs = [0, 2, 3, 4, 5, 6, 8]
        # A dirty trick to prevent a bug regarding a slightly different German preprocessing. 
        if sample == 'German': 
            fmin = (4, 8,  8,  10, 13, 20, 30)
            fmax = (8, 13, 10, 13, 20, 30, 45)
            freq_idcs = [0, 1, 2, 3, 4, 5, 6]
          
        for freq_idx in freq_idcs:
            # Another dirty trick of the same kind, see above. 
            if sample == 'German': 
                matrix_offset = 7
            else:
                matrix_offset = 10
            freq_matrices = matrices[freq_idx::matrix_offset]
            
            # Outlier correction.
            mean = np.mean(freq_matrices)
            std = np.std(freq_matrices)
            mean_weights = np.mean(freq_matrices, axis = 1)
            cond = np.abs(mean_weights - mean) < 3 * std
            freq_matrices = freq_matrices[cond]
            intel_idx = set((np.argwhere(cond)).flatten())
            intel_arr = complete_intel_arr[np.array(list(intel_idx))]
            age_arr = complete_age_arr[np.array(list(intel_idx))]
            print(f'{len(complete_intel_arr)-len(intel_arr)} participants were discarded due to the outlier corerction.')
            
            behav_pred_pos, behav_pred_neg, res_posedges, res_negedges, \
                all_posedges, all_negedges = CPM_fixed.KFold_validation(x = freq_matrices, y = intel_arr, corr = corr_mode,
                                                                        age = age_arr, robustRegression=False,
                                                                        weighted=False, p_threshold=p_threshold,
                                                                        k = len(intel_arr), order=CPM_order)

            os.makedirs(res_dir, exist_ok=True)
            if not np.any(np.isnan(behav_pred_pos)):
                np.save(f'{res_dir}pos_pred_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', 
                        behav_pred_pos)
                np.save(f'{res_dir}posedges_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', res_posedges)
                np.save(f'{res_dir}all_posedges_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', 
                        all_posedges)
                # print(behav_pred_pos)
                print(f'The R2 of pos edges, {fmin[freq_idx]}-{fmax[freq_idx]} Hz: {r2_score(intel_arr, behav_pred_pos)}')
            if not np.any(np.isnan(behav_pred_neg)):                
                np.save(f'{res_dir}neg_pred_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', 
                        behav_pred_neg)
                np.save(f'{res_dir}negedges_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', res_negedges)
                np.save(f'{res_dir}all_negedges_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', 
                        all_negedges)
                print(f'The R2 of neg edges, {fmin[freq_idx]}-{fmax[freq_idx]} Hz: {r2_score(intel_arr, behav_pred_neg)}')
            
            np.save(f'{res_dir}memory_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', 
                    intel_arr)            
               
    stop = time.time()
    print(stop - start)

if __name__ == '__main__':
    main()