# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 19:10:26 2025

@author: Admin
"""

# This file preforms the lesioning procedure. 

import numpy as np
import intel_utils
from intel_utils import parse_result_file
import os
import glob
import CPM_fixed
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from openpyxl import load_workbook


# Anton did his own multiple comparison correction and his results are different, so here we are.
def parse_Anton_results(res_dir, der_res_dir):
    wb = load_workbook(f'{der_res_dir}\\final_results_corrected.xlsx')
    ws = wb.active
    results = []
    for i, row in enumerate(ws.values):
        if i == 0: continue # Skip the header. 
        p_threshold = row[0]
        CPM_order = row[1]
        atlas = row[2]
        dataset = row[3]
        method = row[4]
        corr_sign = row[5]
        freq = row[6]
        r = float(row[7])
        p = float(row[8])
        corrected_p = float(row[9])
        corr_mode = 'partial'
        
        behav = np.load(f'{res_dir}\\{method}\\{atlas} {dataset}\\{corr_mode}\\{CPM_order}\\{p_threshold}\\memory_{freq}.npy')
        pred = np.load(f'{res_dir}\\{method}\\{atlas} {dataset}\\{corr_mode}\\{CPM_order}\\{p_threshold}\\{corr_sign}_pred_{freq}.npy')
        r2 = r2_score(behav, pred)
        mae = mean_absolute_error(behav, pred)
        rmse = root_mean_squared_error(behav, pred)
        result = intel_utils.Result(p_threshold = p_threshold, CPM_order = CPM_order, corr_mode = corr_mode, 
                        atlas = atlas, dataset = dataset, method = method, 
                        corr_sign = corr_sign, freq = freq, r = r, 
                        p = p, corrected_p = corrected_p, r2 = r2, 
                        behav = behav, pred = pred, mae = mae, rmse = rmse)
        results.append(result)
    return results

def main():
    res_dir = 'H:\\Work\\Intelligence\\Results\\CPM results single LOO'    
    der_res_dir = 'H:\\Work\\Intelligence\\Results\\Derivative results'
    
    cond = lambda result: result.corrected_p < 0.05 and result.r > 0 and \
        result.freq != '8-13 Hz'
    results = list(filter(cond, parse_Anton_results(res_dir, der_res_dir)))
    print(results[0])
    # print(list([result.dataset for result in results]))
    # return
    for i, result in enumerate(results[:1]):
        print(f'# {i}------------------------')
        print(result)
        intel_test = 'LPS' # WST, LPS or RWT; applicable to German sample only
        if result.dataset == 'GermanLPS': result.dataset = 'German'
        if result.dataset != 'German': intel_test = ''
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
        
        
        intel_method = intel_method_dict[result.dataset + intel_test]
        age_method = age_method_dict[result.dataset]
        sex_method = sex_method_dict[result.dataset]
        
        behav = intel_method()
        age = age_method()
        sex = sex_method()
        
        conn_dir = f'Matrices\\{result.method}\\{result.atlas}\\{result.dataset}\\'
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
        
        freq_idcs_dict = {'4-8 Hz': 0, '8-13 Hz': 2, '8-10 Hz': 3, '10-13 Hz': 4,
                          '13-20 Hz': 5, '20-30 Hz': 6, '30-45 Hz': 8}
        # A technical debt.
        # fmin = (4, 4,  8,  8,  10, 13, 20, 30, 30, 4)
        # fmax = (8, 30, 13, 10, 13, 20, 30, 40, 45, 45)
        freq_idcs = [freq_idcs_dict[result.freq]]
        # A debt also, but a little less.
        # fmin = (4, 8,  8,  10, 13, 20, 30)
        # fmax = (8, 13, 10, 13, 20, 30, 45)
        if result.dataset == 'German': 
            # A dirty trick to prevent a bug regarding a slightly different German preprocessing. 
            freq_idcs_dict = {'4-8 Hz': 0, '8-13 Hz': 1, '8-10 Hz': 2, '10-13 Hz': 3,
                              '13-20 Hz': 4, '20-30 Hz': 5, '30-45 Hz': 6}
            freq_idcs = [freq_idcs_dict[result.freq]]
          
        for freq_idx in freq_idcs:
            # Another dirty trick of the same kind, see above. 
            if result.dataset == 'German': 
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
        
        edge_dir = f'{res_dir}\\{result.method}\{result.atlas} {result.dataset}{intel_test}\\{result.corr_mode}\{result.CPM_order}\\{result.p_threshold}\\{result.corr_sign}edges_{result.freq}.npy'
        edges = np.load(edge_dir)
        idcs = np.argwhere(edges)
        print(idcs)
        for idx in idcs:
            lesion_res_dir = 'H:\\Work\\Intelligence\\Results\\CPM results single LOO lesion\\' +\
            f'{result.method}\\{result.atlas} {result.dataset}{intel_test}\\partial\\{result.CPM_order}\\' +\
                f'{result.p_threshold}\\{result.corr_sign}_lesion_{idx}'
            os.makedirs(lesion_res_dir, exist_ok=True)
            lesioned_freq_matrices = np.delete(freq_matrices, idx, axis=1)
            behav_pred_pos, behav_pred_neg, res_posedges, res_negedges, \
                all_posedges, all_negedges = CPM_fixed.KFold_validation(x = lesioned_freq_matrices, y = intel_arr, corr = result.corr_mode,
                                                                        age = age_arr, robustRegression=False,
                                                                        weighted=False, p_threshold=float(result.p_threshold.split(' ')[0]),
                                                                        k = len(intel_arr), order=int(result.CPM_order.split(' ')[0]))
            if not np.any(np.isnan(behav_pred_pos)):
                np.save(f'{lesion_res_dir}\\pos_pred_{result.freq}', 
                        behav_pred_pos)
                np.save(f'{lesion_res_dir}\\posedges_{result.freq}', res_posedges)
                np.save(f'{lesion_res_dir}\\all_posedges_{result.freq}', 
                        all_posedges)
                print(f'The R2 of pos edges, {result.freq}, {idx} edge: {r2_score(intel_arr, behav_pred_pos)}')
            print(behav_pred_neg)
            if not np.any(np.isnan(behav_pred_neg)):                
                np.save(f'{lesion_res_dir}\\neg_pred_{result.freq}', 
                        behav_pred_neg)
                np.save(f'{lesion_res_dir}\\negedges_{result.freq}', res_negedges)
                np.save(f'{lesion_res_dir}\\all_negedges_{result.freq}', 
                        all_negedges)
                print(f'The R2 of neg edges, {result.freq}, {idx} edge: {r2_score(intel_arr, behav_pred_neg)}')
            
            np.save(f'{lesion_res_dir}\\memory_{result.freq}', 
                    intel_arr)            
            
            
if __name__ == '__main__':
    main()