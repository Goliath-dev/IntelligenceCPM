# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:56:03 2025

@author: Admin
"""

# This script performs all-to-all external validation. 

import intel_utils
from intel_utils import Result
import numpy as np
import CPM_fixed
import scipy as sp
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from openpyxl import load_workbook
from itertools import combinations

def extract_identifying_fields(instance):
    return (instance.p_threshold, instance.CPM_order, instance.atlas, 
            instance.method, instance.corr_sign, instance.freq)

def predict(train_results, test_dataset):
    for result in train_results:       
        edge_dir = f'{res_dir}\\{result.method}\\{result.atlas} {result.dataset}\\{result.corr_mode}\\{result.CPM_order}\\{result.p_threshold}\\{result.corr_sign}edges_{result.freq}.npy'
        # The matrices are taken from double LOO in order to simplify the procedure - 
        # they were saved in the double LOO, but not in the single one which was 
        # developed earlier (when I was young and didn't think about that). The matrice
        # sets are obviously the same as they do not depend on the LOO type. 
        matrices_dir = f'{res_dir.replace("CPM results single LOO", "CPM results double LOO")}\\{result.method}\\{result.atlas} {result.dataset}\\{result.corr_mode}\\{result.CPM_order}\\{result.p_threshold}\\complete_matrices_{result.freq}.npy'
        behav_dir = f'{res_dir}\\{result.method}\\{result.atlas} {result.dataset}\\{result.corr_mode}\\{result.CPM_order}\\{result.p_threshold}\\memory_{result.freq}.npy'
        matrices = np.load(matrices_dir)
        edges = np.load(edge_dir)
        behav = np.load(behav_dir)
        test_matrices = np.load(matrices_dir.replace(result.dataset, test_dataset))
        test_behav = np.load(behav_dir.replace(result.dataset, test_dataset))

        if np.any(edges):
            order = int(result.CPM_order.split(' ')[0])
            FC_sum = matrices.T[edges, :].sum(axis=0)
            fit = np.polyfit(FC_sum, behav, order)
            test_FC_sum = test_matrices.T[edges, :].sum(axis=0)
            pred = CPM_fixed.poly_generator(order, fit)(test_FC_sum)
            r, p = sp.stats.pearsonr(test_behav, pred)
            # if r > 0 and p < 0.05:
            print(f'{result.method}, {result.freq}, {result.atlas}, {result.p_threshold}, {result.CPM_order}')
            print(f'r = {r}, p = {p}')
                
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

res_dir = 'H:\\Work\\Intelligence\\Results\\CPM results single LOO'    
der_res_dir = 'H:\\Work\\Intelligence\\Results\\Derivative results'
results = list(filter(lambda result: result.corrected_p < 0.05 and result.r > 0, parse_Anton_results(res_dir, der_res_dir)))
datasets = set([result.dataset for result in results])
res_by_datasets = {dataset: list(filter(lambda result: result.dataset == dataset, results)) for dataset in datasets}

for first, second in combinations(datasets, 2):
    first_set = {extract_identifying_fields(instance) for instance in res_by_datasets[first]}
    second_set = {extract_identifying_fields(instance) for instance in res_by_datasets[second]}
    common_fields = first_set.intersection(second_set)
    if len(common_fields) != 0:
        print(first, second)
        first_common = [instance for instance in res_by_datasets[first] if extract_identifying_fields(instance) in common_fields]
        second_common = [instance for instance in res_by_datasets[second] if extract_identifying_fields(instance) in common_fields]
        predict(first_common, second)
        predict(second_common, first)
    
