# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 17:56:56 2023

@author: Admin
"""

# This is a base CPM script for this research. Basically, it just contains
# code from CBMPCBPM.py and CPM_German.py, which, in turn, mostly repeat 
# each other, so no need to keep two versions of what is supposed to be
# one and the same. 
# This file applies connectome-based predictive modelling (CPM) to the data of
# German, Cuban and Chelyabinsk samples. 

import CPM_fixed
import intel_utils
import glob
import numpy as np
from CPMPlotting import CPMPlot, CPMTable
from sklearn.metrics import r2_score
import seaborn as sns
import pandas as pd
import time
import os
    
start = time.time()

# Parameters. 
CPM_order = 1
p_threshold = 0.001
method = 'imCoh' # PLV, wPLI or imCoh
sample = 'Chel' # Cuba, Chel, German
corr_mode = 'partial' # corr or partial
atlas = 'DK' # Destrieux or DK
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
sex_method_dict = {'Chel': intel_utils.get_Raven_sex,
                   'Cuba': intel_utils.get_WAIS_sex,
                   'German': intel_utils.get_German_sex}

intel_method = intel_method_dict[sample + intel_test]
age_method = age_method_dict[sample]
sex_method = sex_method_dict[sample]

res_dir = f'Results\\CPM results {validation}\\{method}\\{atlas} {sample}{intel_test}\\{corr_mode}\\{CPM_order} order\\{p_threshold} p_value\\'
img_dir = f'Results\\CPM plots {validation}\\{method}\\{atlas} {sample}{intel_test}\\{corr_mode}\\{CPM_order} order\\{p_threshold} p_value\\'

behav = intel_method()
age = age_method()
sex = sex_method()

conn_dir = f'Matrices\\{method}\\{atlas}\\{sample}\\'
files = glob.glob(conn_dir + '*.npy')

label_dir = f'Labels\\{atlas}\\'
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

# Several last regions are meaningless, so drop them. Their count depends on
# the exact atlas, so here we go.
ROI_offset = 2 if atlas == 'Destrieux' else 1
ROI_labels = np.load(label_dir + 'CBM00001_labels.npy', allow_pickle=True)[:-ROI_offset]

cpm_table = CPMTable()

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
      
    cpm_table.add_col(fmin[freq_idx], fmax[freq_idx])
    
    if validation == 'LOO':
        pos_pred, neg_pred, \
        posedges, negedges, \
        all_posedges, all_negedges = CPM_fixed.LOO_validation(freq_matrices.T, intel_arr, 
                                                      CPM_order, weighted = False,
                                                      corr = corr_mode, covar = age_arr,
                                                      p_threshold=p_threshold)
    elif validation == 'k-fold':
        pos_pred, neg_pred, \
        posedges, negedges, \
        all_posedges, all_negedges = CPM_fixed.k_fold_validation(freq_matrices.T, intel_arr, 
                                                      CPM_order, weighted = False,
                                                      corr = corr_mode, covar = age_arr,
                                                      p_threshold=p_threshold)
    
    os.makedirs(res_dir, exist_ok=True)
  
    cpm_plot = CPMPlot(ROI_labels, fmin[freq_idx], fmax[freq_idx])
    
    # Prediction can return NaN due to absence of valueable edges, so leave only
    # those LOO runs in which CPM worked correctly.
    correct_pos_idcs = np.logical_not(np.isnan(pos_pred))
    correct_neg_idcs = np.logical_not(np.isnan(neg_pred))
    correct_pos_pred = pos_pred[correct_pos_idcs]
    correct_neg_pred = neg_pred[correct_neg_idcs]
    
    success_pos_rate = len(correct_pos_pred) / len(pos_pred)
    success_neg_rate = len(correct_neg_pred) / len(neg_pred)
    
    print(f'Positive success rate is {success_pos_rate} at [{fmin[freq_idx]} - {fmax[freq_idx]}] Hz band.')
    print(f'Negative success rate is {success_neg_rate} at [{fmin[freq_idx]} - {fmax[freq_idx]}] Hz band.')
          
    
    if success_pos_rate > 0.95:
        cpm_plot.setup_pos_scatter(intel_arr[correct_pos_idcs], correct_pos_pred)
        cpm_plot.set_pos_matrix(posedges)
        cpm_table.update_pos_table(intel_arr[correct_pos_idcs], correct_pos_pred)
        np.save(f'{res_dir}pos_pred_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', 
                correct_pos_pred)
        np.save(f'{res_dir}posedges_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', posedges)
        np.save(f'{res_dir}all_posedges_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', 
                all_posedges)
        np.save(f'{res_dir}intel_arr_pos_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', 
                intel_arr[correct_pos_idcs])
        
        # "Lesioned" CPM, that is, if we achieved a significant result (R2 > 0.05), then 
        # we remove one valuable edge at a time in order to assess how the removed edge
        # influence the quality prediction. 
        r2 = r2_score(intel_arr[correct_pos_idcs], correct_pos_pred) # Here and below TODO: decompose for God's sake. 
        if r2 >= 0.05:
            print('R2 is over 0.05, starting lesion check...')
            idcs = np.argwhere(posedges)
            lesion_r2s = [r2]
            edge_names = ['Nothing removed']
            for idx in idcs:
                lesion_freq_matrices = np.delete(freq_matrices, idx, axis=1);
                lesion_pos_pred, lesion_neg_pred, \
                lesion_posedges, lesion_negedges, \
                lesion_all_posedges, lesion_all_negedges = CPM_fixed.LOO_validation(lesion_freq_matrices.T, intel_arr, 
                                                              CPM_order, weighted = False,
                                                              corr = corr_mode, covar = age_arr,
                                                              p_threshold=p_threshold)
                
                lesion_correct_pos_idcs = np.logical_not(np.isnan(lesion_pos_pred))
                lesion_correct_pos_pred = lesion_pos_pred[lesion_correct_pos_idcs]
                
                lesion_success_pos_rate = len(lesion_correct_pos_pred) / len(lesion_pos_pred)
                
                lesion_r2s.append(r2_score(intel_arr[lesion_correct_pos_idcs], lesion_correct_pos_pred))
                tr_indices = np.triu_indices(len(ROI_labels), k=1)
                edge_names.append(f'{ROI_labels[tr_indices[0][idx]][0].name} to \n {ROI_labels[tr_indices[1][idx]][0].name}')
                
                if lesion_success_pos_rate > 0.95:
                    plot_edges = np.insert(lesion_posedges, idx, False)
                    lesion_cpm_plot = CPMPlot(ROI_labels, fmin[freq_idx], fmax[freq_idx])
                    lesion_cpm_plot.setup_pos_scatter(intel_arr[lesion_correct_pos_idcs], lesion_correct_pos_pred)
                    lesion_cpm_plot.set_pos_matrix(plot_edges)
                    # lesion_cpm_table.update_pos_table(intel_arr[lesion_correct_pos_idcs], lesion_correct_pos_pred)
                    os.makedirs(f'{res_dir}lesion_{idx}/pos_pred_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', exist_ok=True)
                    os.makedirs(f'{res_dir}lesion_{idx}/posedges_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', exist_ok=True)
                    os.makedirs(f'{res_dir}lesion_{idx}/all_posedges_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', exist_ok=True)
                    os.makedirs(f'{res_dir}lesion_{idx}/intel_arr_pos_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', exist_ok=True)
                    np.save(f'{res_dir}lesion_{idx}/pos_pred_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', 
                            lesion_correct_pos_pred)
                    np.save(f'{res_dir}lesion_{idx}/posedges_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', plot_edges)
                    np.save(f'{res_dir}lesion_{idx}/all_posedges_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', 
                            lesion_all_posedges)
                    np.save(f'{res_dir}lesion_{idx}/intel_arr_pos_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', 
                            intel_arr[lesion_correct_pos_idcs])
                    lesion_cpm_plot.plot_quad_plot(f'{img_dir}lesion_pos_{idx}')
            plot_data =  pd.DataFrame(np.array([lesion_r2s, edge_names], dtype=object).T, columns = ['R²', 'Removed valuable edges'])
            lesion_aggr_ax = sns.catplot(plot_data, x = 'Removed valuable edges', y = 'R²', kind='point')
            lesion_aggr_ax.tick_params(axis='x', rotation=90)
            lesion_aggr_ax.figure.savefig(img_dir + f'[{fmin[freq_idx]}-{fmax[freq_idx]}] Hz lesion_aggr_pos', bbox_inches='tight')        
    else:
        print(f'Success rate of positive edges during LOO-validation at [{fmin[freq_idx]} - {fmax[freq_idx]}] Hz band is too low.')
    
    if success_neg_rate > 0.95:
        cpm_plot.setup_neg_scatter(intel_arr[correct_neg_idcs], correct_neg_pred)
        cpm_plot.set_neg_matrix(negedges)
        cpm_table.update_neg_table(intel_arr[correct_neg_idcs], correct_neg_pred)
        np.save(f'{res_dir}neg_pred_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', correct_neg_pred)
        np.save(f'{res_dir}negedges_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', negedges)
        np.save(f'{res_dir}all_negedges_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', 
                all_negedges)
        np.save(f'{res_dir}intel_arr_neg_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', 
                intel_arr[correct_neg_idcs])
        
        r2 = r2_score(intel_arr[correct_neg_idcs], correct_neg_pred)
        if r2 >= 0.05:
            print('R2 is over 0.05, starting lesion check...')
            idcs = np.argwhere(negedges)
            lesion_r2s = [r2]
            edge_names = ['Nothing removed']
            for idx in idcs:
                lesion_freq_matrices = np.delete(freq_matrices, idx, axis=1);
                lesion_pos_pred, lesion_neg_pred, \
                lesion_posedges, lesion_negedges, \
                lesion_all_posedges, lesion_all_negedges = CPM_fixed.LOO_validation(lesion_freq_matrices.T, intel_arr, 
                                                              CPM_order, weighted = False,
                                                              corr = corr_mode, covar = age_arr,
                                                              p_threshold=p_threshold)
                
                lesion_correct_neg_idcs = np.logical_not(np.isnan(lesion_neg_pred))
                lesion_correct_neg_pred = lesion_neg_pred[lesion_correct_neg_idcs]
                
                lesion_success_neg_rate = len(lesion_correct_neg_pred) / len(lesion_neg_pred)
                
                lesion_r2s.append(r2_score(intel_arr[lesion_correct_neg_idcs], lesion_correct_neg_pred))
                tr_indices = np.triu_indices(len(ROI_labels), k=1)
                edge_names.append(f'{ROI_labels[tr_indices[0][idx]][0].name} to \n {ROI_labels[tr_indices[1][idx]][0].name}')
                
                if lesion_success_neg_rate > 0.95:
                    plot_edges = np.insert(lesion_negedges, idx, False)
                    lesion_cpm_plot = CPMPlot(ROI_labels, fmin[freq_idx], fmax[freq_idx])
                    lesion_cpm_plot.setup_neg_scatter(intel_arr[lesion_correct_neg_idcs], lesion_correct_neg_pred)
                    lesion_cpm_plot.set_neg_matrix(plot_edges)
                    # lesion_cpm_table.update_pos_table(intel_arr[lesion_correct_pos_idcs], lesion_correct_pos_pred)
                    os.makedirs(f'{res_dir}lesion_{idx}/neg_pred_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', exist_ok=True)
                    os.makedirs(f'{res_dir}lesion_{idx}/negedges_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', exist_ok=True)
                    os.makedirs(f'{res_dir}lesion_{idx}/all_negedges_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', exist_ok=True)
                    os.makedirs(f'{res_dir}lesion_{idx}/intel_arr_neg_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', exist_ok=True)
                    np.save(f'{res_dir}lesion_{idx}/neg_pred_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', 
                            lesion_correct_neg_pred)
                    np.save(f'{res_dir}lesion_{idx}/negedges_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', plot_edges)
                    np.save(f'{res_dir}lesion_{idx}/all_negedges_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', 
                            lesion_all_negedges)
                    np.save(f'{res_dir}lesion_{idx}/intel_arr_neg_{fmin[freq_idx]}-{fmax[freq_idx]} Hz', 
                            intel_arr[lesion_correct_neg_idcs])
                    lesion_cpm_plot.plot_quad_plot(f'{img_dir}lesion_neg_{idx}')
            plot_data =  pd.DataFrame(np.array([lesion_r2s, edge_names], dtype=object).T, columns = ['R²', 'Removed valuable edges'])
            lesion_aggr_ax = sns.catplot(plot_data, x = 'Removed valuable edges', y = 'R²', kind='point')
            lesion_aggr_ax.tick_params(axis='x', rotation=90)
            lesion_aggr_ax.figure.savefig(img_dir + f'[{fmin[freq_idx]}-{fmax[freq_idx]}] Hz lesion_aggr_neg', bbox_inches='tight')        
        
    else:
        print(f'Success rate of negative edges during LOO-validation at [{fmin[freq_idx]} - {fmax[freq_idx]}] Hz band is too low.')
    
    cpm_plot.plot_quad_plot(img_dir)

cpm_table.save_table(img_dir)
    
    

stop = time.time()
print(stop - start)