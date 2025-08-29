# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 13:20:55 2025

@author: Admin
"""

# This script agregates lesion results. 

from intel_utils import Result
import glob
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from intel_utils import steiger_res_from_res
from statsmodels.stats.multitest import multipletests
import csv
from dataclasses import fields
import os
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from openpyxl import load_workbook
from scipy.optimize import curve_fit
from itertools import combinations

def compare_if_same_but_order(res1, res2):
    return res1.p_threshold == res2.p_threshold and res1.atlas == res2.atlas and \
        res1.dataset == res2.dataset and res1.method == res2.method and \
            res1.corr_sign == res2.corr_sign and res1.freq == res2.freq
    
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
        result = Result(p_threshold = p_threshold, CPM_order = CPM_order, corr_mode = corr_mode, 
                        atlas = atlas, dataset = dataset, method = method, 
                        corr_sign = corr_sign, freq = freq, r = r, 
                        p = p, corrected_p = corrected_p, r2 = r2, 
                        behav = behav, pred = pred, mae = mae, rmse = rmse)
        results.append(result)
    return results

def TOST_steiger_z(r_jk, r_jh, r_kh, N, epsilon):
    left_side = steiger_z(r_jk, r_jh - epsilon / 2, r_kh, N)
    right_side = steiger_z(r_jk, r_jh + epsilon / 2, r_kh, N)
    left_side_z = left_side[0]
    right_side_z = right_side[0]
    left_side_p = sp.stats.norm.cdf(left_side_z)
    right_side_p = 1 - sp.stats.norm.cdf(right_side_z)
    # print(f'Left-side p is {left_side_p}, left-side z is {left_side_z}, right-side p is {right_side_p}, right-side z is {right_side_z}, r_jk = {r_jk}, r_jh = {r_jh}.')
    return min(left_side_p, right_side_p)

def steiger_z(r_jk, r_jh, r_kh, N):
    """
    Perform Steiger's Z-test for comparing two dependent correlations sharing a common index.
    
    Parameters:
    - r_jk, r_jh: Sample correlations to compare (sharing index j).
    - r_kh: Correlation between variables k and h.
    - N: Sample size.
    
    Returns:
    - Z: Test statistic.
    - p_value: Two-tailed p-value.
    """
    # Fisher's z-transformation
    z_jk = 0.5 * np.log((1 + r_jk) / (1 - r_jk))
    z_jh = 0.5 * np.log((1 + r_jh) / (1 - r_jh))
    
    # Pooled correlation under H0
    r_bar = (r_jk + r_jh) / 2
    
    # Compute ψ using Equation (3) with pooled r
    term1 = r_kh * (1 - 2 * r_bar**2)
    term2 = 0.5 * r_bar**2 * (1 - 2 * r_bar**2 - r_kh**2)
    psi = term1 - term2
    
    # Compute c (covariance term)
    denominator_c = (1 - r_bar**2)**2
    c = psi / denominator_c
    
    # Compute test statistic (Equation 14)
    numerator = (z_jk - z_jh) * np.sqrt(N - 3)
    denominator = np.sqrt(2 - 2 * c)
    Z = numerator / denominator
    
    p_value = 2 * (1 - sp.stats.norm.cdf(abs(Z)))
    
    return Z, p_value    
   
def edge_name_from_idx(idx, atlas):
    if np.isnan(idx): 
        return 'Whole brain model'
    label_dir = f'Labels\\{atlas}\\'
    ROI_offset = 2 if atlas == 'Destrieux' else 1
    ROI_labels = np.load(label_dir + 'CBM00001_labels.npy', allow_pickle=True)[:-ROI_offset]
    tr_indices = np.triu_indices(len(ROI_labels), k=1)
    edge_name = f'{ROI_labels[tr_indices[0][idx]].name} to \n {ROI_labels[tr_indices[1][idx]].name}'
    return edge_name
 
def calc_steiger_results(result, r_arr):
    rs = [r[0] for r in r_arr]
    ps = [r[3] for r in r_arr]
    edge_names = [edge_name_from_idx(r[2], result.atlas) for r in r_arr]
    steiger_results = []
    for i, edge_name in enumerate(edge_names):
        if i == 0: continue
        z, steiger_p = steiger_z(result.r, rs[i], r_arr[i][1], len(result.pred))
        # steiger_p = TOST_steiger_z(result.r, rs[i], r_arr[i][1], len(result.pred), epsilon = 0.025)
        steiger_result = steiger_res_from_res(result)
        steiger_result.steiger_z = z
        steiger_result.steiger_p = steiger_p
        steiger_result.excluded_edge = edge_name
        steiger_result.pearson_r = rs[i]
        steiger_result.pearson_p = ps[i]
        steiger_results.append(steiger_result)
    return steiger_results

def create_steiger_matrices(result, steiger_res, idcs, save_dir):
    # steiger_ps = np.array([res.corrected_steiger_p for res in steiger_res])
    # A kludge to perform a multiple comparison correction in another way.
    steiger_ps = np.array([res.steiger_p for res in steiger_res])
    if np.all(steiger_ps > 0.05): return
    # Extract only indices o significant edges. 
    # sign_idx_p_arr = [(idx, p) for idx, p in zip(idcs, steiger_ps) if p <= 0.05]
    # A kludge to perform a multiple comparison correction in another way.
    sign_idx_p_arr = [(idx, p * len(idcs)) for idx, p in zip(idcs, steiger_ps) if p <= 0.05 / len(idcs)]
    sign_idcs = [el[0] for el in sign_idx_p_arr]
    sign_ps = [el[1] for el in sign_idx_p_arr]
    N = 148 if result.atlas == 'Destrieux' else 68
    tr_indices = np.triu_indices(N, k=1)
    matrix = np.zeros((N, N))
    for idx, p in zip(sign_idcs, sign_ps):
        i = tr_indices[0][idx]
        j = tr_indices[1][idx]
        matrix[i, j] = 1 / p
        matrix[j, i] = 1 / p
    file_name = f'{result.p_threshold}_{result.CPM_order}_{result.atlas}_{result.dataset}_{result.method}_{result.corr_sign}_{result.freq}'
    os.makedirs(save_dir, exist_ok=True)
    with open(f'{save_dir}\\{file_name}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for row in matrix:
            writer.writerow(row)
    
                
def plot_result(result, r_arr, img_dir):
    rs = [r[0] for r in r_arr]
    edge_names = [edge_name_from_idx(r[2], result.atlas) for r in r_arr]
    plot_data =  pd.DataFrame(np.array([rs, edge_names], dtype=object).T, columns = ['Correlation coefficient', 'Removed valuable edges'])
    fig, ax = plt.subplots(figsize = (8, 4))
    ax = sns.barplot(data=plot_data, x = 'Removed valuable edges', y = 'Correlation coefficient', linewidth = 0.4, dodge = False, saturation=0.4, ax = ax)
    ax.set_xlabel(ax.get_xlabel(), fontsize = 16)
    ax.set_ylabel(ax.get_ylabel(), fontsize = 16)
    ax.tick_params(axis = 'x', labelsize = 16)
    ax.tick_params(axis = 'y', labelsize = 16)
    ax.tick_params(axis='x', rotation=90)
    labels = [str(round(r, 2)) for r in rs]
    N = len(rs) - 1
    for i, label in enumerate(labels):
        if i == 0: continue
        z, steiger_p = steiger_z(result.r, rs[i], r_arr[i][1], len(result.pred))
        steiger_p = steiger_p * N
        # steiger_p = TOST_steiger_z(result.r, rs[i], r_arr[i][1], len(result.pred), epsilon = 0.025)
        # print(TOST_steiger_z(result.r, rs[i], r_arr[i][1], len(result.pred), epsilon = 0.05))
        # print((steiger_p, result.r, rs[i], r_arr[i][1]))
        text = ''
        if steiger_p < 0.05:
            if (steiger_p <= 0.05) and (steiger_p > 0.01): 
                text = '*'
            elif (steiger_p <= 0.01) and (steiger_p > 0.001):
                text = '**'
            elif steiger_p <= 0.001:
                text = '***'
        labels[i] += text
    ax.bar_label(ax.containers[0], fontsize=16, labels = labels)
    ax.set_title(f'{result.method}, {result.atlas}, {result.dataset}, {result.CPM_order}, {result.p_threshold}, {result.corr_sign}, {result.freq}')
    
    h = 1.2
    ax.set_ylim(top = (h) * max(rs))
    ax.figure.savefig(f'{img_dir}\\{result.method}_{result.atlas}_{result.dataset}_{result.CPM_order}_{result.p_threshold}_{result.corr_sign}_{result.freq}_lesion_aggr.png', bbox_inches='tight')

res_dir = 'H:\\Work\\Intelligence\\Results\\CPM results single LOO'
der_res_dir = 'H:\\Work\\Intelligence\\Results\\Derivative results'
img_dir = 'Results\\Imgs for article\\Lesion aggr'

# cond = lambda result: result.corrected_p < 0.05 and result.r > 0 and \
#     result.freq != '8-13 Hz'
# results = list(filter(cond, parse_result_file(res_dir, der_res_dir)))
cond = lambda result: result.corrected_p < 0.05 and result.r > 0 and \
    result.freq != '8-13 Hz'
results = list(filter(cond, parse_Anton_results(res_dir, der_res_dir)))

steiger_res_arr = [] # This one is needed for the agregation file. This is a plain list of all steiger results.
steiger_res_aux_arr = [] # This one is needed for the glass brain. This is a list of lists, where every element is a list of steiger results corresponding to a main result. 
for result in results:
    r_arr = [(result.r, 1, np.nan, np.nan)]
    lesion_res_dir = 'H:\\Work\\Intelligence\\Results\\CPM results single LOO lesion\\' +\
    f'{result.method}\\{result.atlas} {result.dataset}\\partial\\{result.CPM_order}\\' +\
        f'{result.p_threshold}\\{result.corr_sign}_lesion_*'
    lesions = glob.glob(lesion_res_dir)
    idcs_arr = []
    for lesion in lesions:
        edge_idx = int(lesion.split('_')[-1][1:-1])
        idcs_arr.append(edge_idx)
        # There might be cases when for given edge index there is no result in this frequency.
        # Or, rather, for a given frequency there's some edges, then for another frequency 
        # with the same parameters (e. g., in the same directory) there are some other edges, 
        # and the directory contains the union of the edge sets and therefore the idcs_arr
        # will contain all these edges. But the folders corresponding to these edges will
        # contain only "their own" frequencies, which result in a FileNotFound error. 
        # To prevent this, here's a workaround.
        if not os.path.isfile(f'{lesion}\\memory_{result.freq}.npy') \
            or not os.path.isfile(f'{lesion}\\{result.corr_sign}_pred_{result.freq}.npy'): continue
        behav = np.load(f'{lesion}\\memory_{result.freq}.npy')
        pred = np.load(f'{lesion}\\{result.corr_sign}_pred_{result.freq}.npy')
        r, p = sp.stats.pearsonr(behav, pred)
        r_arr.append((r, sp.stats.pearsonr(pred, result.pred)[0], edge_idx, p))
    plot_result(result, r_arr, img_dir)
    steiger_res = calc_steiger_results(result, r_arr)
    steiger_res_aux_arr.append(steiger_res)
    steiger_res_arr.extend(steiger_res)

# Gotta change that later, we perform correction not over all results now, but inside 
# specific result between lesioned variants instead. For now I have a kludge in the 
# create_steiger_matrices function.
steiger_p_values = [steiger_result.steiger_p for steiger_result in steiger_res_arr]
pearson_p_values = [steiger_result.pearson_p for steiger_result in steiger_res_arr]
steiger_multiple_correction_res = multipletests(steiger_p_values, alpha = 0.05, method = 'bonferroni')
pearson_multiple_correction_res = multipletests(pearson_p_values, alpha = 0.05, method = 'bonferroni')

# Write a file with agregated Steiger results. 
# with open(f'{der_res_dir}\\steiger_results.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile, delimiter=',')
#     field_names = list([field.name for field in fields(steiger_res_arr[0])])
#     writer.writerow(field_names)
#     for steiger_result, corrected_steiger_p, corrected_pearson_p in zip(steiger_res_arr, steiger_multiple_correction_res[1], pearson_multiple_correction_res[1]):
#         steiger_result.corrected_steiger_p = corrected_steiger_p
#         steiger_result.corrected_pearson_p = corrected_pearson_p
#         row = [str(getattr(steiger_result, field.name)) for field in fields(steiger_result)]
#         writer.writerow(row)

# Write files with significant (in terms of Steiger's p-value) edges to draw glass brains.  
for result, steiger_res in zip(results, steiger_res_aux_arr):
    lesion_res_dir = 'H:\\Work\\Intelligence\\Results\\CPM results single LOO lesion\\' +\
    f'{result.method}\\{result.atlas} {result.dataset}\\partial\\{result.CPM_order}\\' +\
        f'{result.p_threshold}\\{result.corr_sign}_lesion_*'
    lesions = glob.glob(lesion_res_dir)
    idcs_arr = []
    for lesion in lesions:
        if not os.path.isfile(f'{lesion}\\memory_{result.freq}.npy') \
            or not os.path.isfile(f'{lesion}\\{result.corr_sign}_pred_{result.freq}.npy'): continue
        edge_idx = int(lesion.split('_')[-1][1:-1])
        idcs_arr.append(edge_idx)
    create_steiger_matrices(result, steiger_res, idcs_arr, f'{der_res_dir}\\Glass brain matrices')

# Analyze the dependency of the edge count and prediction difference.
len_arr = []
diff_arr = []
for aux_res, res in zip(steiger_res_aux_arr, results):
    if len(aux_res) == 0: continue
    len_arr.append(len(aux_res))
    diff = res.r - min(aux_res, key = lambda res: res.pearson_r).pearson_r
    diff_arr.append(diff)

a, _ = curve_fit(lambda x, a: a / x, len_arr, diff_arr)
regr = a / np.array(len_arr)
print(sp.stats.pearsonr(regr, np.array(diff_arr)))
plt.show()
plt.scatter(len_arr, diff_arr)

# Find the cases when changing in CPM order resulted in changing the Steiger p-value. 
# Probably I would've done smth better than O(N^2), but not at 22:00 on Sunday. 
# for ((result, steiger_res), (in_result, in_steiger_res)) in combinations(zip(results, steiger_res_aux_arr), 2):
#     if compare_if_same_but_order(result, in_result):
#         # if len(list(filter(lambda res: res.steiger_p < 0.05 / len(steiger_res), steiger_res))) != len(list(filter(lambda res: res.steiger_p < 0.05 / len(in_steiger_res), in_steiger_res))):
#             edge_names = [res.excluded_edge for res in steiger_res if res.steiger_p < 0.05 / len(steiger_res)]
#             in_edge_names = [res.excluded_edge for res in in_steiger_res if res.steiger_p < 0.05 / len(in_steiger_res)]
#             edge_intersection = set(edge_names).intersection(set(in_edge_names))
#             if len(edge_intersection) == len(edge_names) == len(in_edge_names): continue
#             print(f'Difference is {len(list(filter(lambda res: res.steiger_p < 0.05 / len(steiger_res), steiger_res))) - len(list(filter(lambda res: res.steiger_p < 0.05 / len(in_steiger_res), in_steiger_res)))}')
#             print('Result is: ')
#             print(f'{result}')
#             print(f'Steiger p-values of the result is: {[res.steiger_p * len(steiger_res) for res in steiger_res]}')
#             print('Other order result is: ')
#             print(f'{in_result}')
#             print(f'Steiger p-values of the other order is: {[res.steiger_p * len(in_steiger_res) for res in in_steiger_res]}')
#             for res in steiger_res:
#                 if res.excluded_edge in edge_names and not res.excluded_edge in edge_intersection:
#                     corr_res = [_res for _res in in_steiger_res if _res.excluded_edge == res.excluded_edge][0]
#                     print(f'For {res.excluded_edge}:')
#                     print(f'P-value was {res.steiger_p * len(steiger_res)} and became {corr_res.steiger_p * len(in_steiger_res)}')
#                     print(f'Z-value was {res.steiger_z} and became {corr_res.steiger_z}')
#             for in_res in in_steiger_res:
#                 if in_res.excluded_edge in in_edge_names and not in_res.excluded_edge in edge_intersection:
#                     corr_res = [_res for _res in steiger_res if _res.excluded_edge == in_res.excluded_edge][0]
#                     print(f'For {corr_res.excluded_edge}:')
#                     print(f'P-value was {corr_res.steiger_p * len(steiger_res)} and became {in_res.steiger_p * len(in_steiger_res)}')
#                     print(f'Z-value was {corr_res.steiger_z} and became {in_res.steiger_z}')
#             print('------------------------------------------------------')


# Create a table with corrected results in Anton's style (with some fields omitted).
with open(f'{der_res_dir}\\Anton_steiger_results.csv', 'w', newline='') as csvfile:
    A_writer = csv.writer(csvfile, delimiter=',')
    field_names = ['p_threshold', 'CPM_order', 'atlas', 'dataset', 'method', 'corr_sign', 'freq', 'excluded_edge', 'steiger_z', 'p_corrected']
    A_writer.writerow(field_names)
    for steiger_res in steiger_res_aux_arr:
        for res in steiger_res:
            row = [str(getattr(res, field)) for field in field_names if hasattr(res, field)]
            p_corrected = res.steiger_p * len(steiger_res)
            row = row + [str(p_corrected)]
            if p_corrected <= 0.05:
                print(row)
                A_writer.writerow(row)
