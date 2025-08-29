# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 13:10:59 2025

@author: Admin
"""

# This script determines the common valuable edges across datasets, if any.

import glob
from openpyxl import load_workbook
import numpy as np
from intel_utils import ResultEdges, read_csv

res_dir = 'H:\Work\Intelligence\Results\CPM results single LOO'
der_res_dir = 'H:\Work\Intelligence\Results\Derivative results'

edge_files = glob.glob(f'{der_res_dir}\\Edges\\*')

Anton_results = f'{der_res_dir}\\final_results_corrected.xlsx'
wb = load_workbook(Anton_results)
ws = wb.active

result_edges_arr = []
for i, row in enumerate(ws.values):
    if i == 0: continue # Skip the header. 
    p_threshold = row[0]
    CPM_order = row[1]
    atlas = row[2]
    dataset = row[3]
    method = row[4]
    corr_sign = row[5]
    freq = row[6]
    
    edges_filename = f'{method}_{atlas}_{dataset}_{CPM_order}_{p_threshold}_{corr_sign}_{freq}_edges.csv'
    # Did you know how NumPy works with array of strings? You do now.
    valuable_edges = [edge[0] for edge in read_csv(f'{der_res_dir}\\Edges\\{edges_filename}')[1:]]
    result_edges = ResultEdges(p_threshold, CPM_order, 'partial', atlas, dataset, method, corr_sign, freq, valuable_edges)
    result_edges_arr.append(result_edges)

# Search for edge intersection.
# freqs = set(res.freq for res in result_edges_arr)
# for freq in freqs:
# # Condition under which the resuts are filtered.
#     cond = lambda res: res.freq == freq and res.corr_sign == 'neg' and res.atlas == 'Destrieux' and res.CPM_order == '1 order' and res.p_threshold == '0.01 p_value'
    
#     filtered_result_edges = list(filter(cond, result_edges_arr))
#     if len(filtered_result_edges) != 0: 
#         all_edges = [set(res.valuable_edges) for res in filtered_result_edges]
#         common_edges = set.intersection(*all_edges)
#         print(f'For the {freq} band the common edges are: \n {common_edges}')
#         print(f'The datasets are: {set([res.dataset for res in filtered_result_edges])}')
#         print(f'The number of results is {len(filtered_result_edges)}')
#         print('----------------------------------------------------------------------------')
        
# Define the number of inclusion of every valuable edge, i. e., in how many results the edge is appeared. 
# If there are several results with the same parameters except the CPM order, it'll have the same set of valuable edges for all orders.
# E. g., there are three results at imCoh, Destrieux, Chel, 0.001 p-value threshold, neg, 8-10 Hz: for 1st, 2nd and 3rd orders of CPM, respectfully.
# All these results will have the same set of valuable edges, as the order of CPM does not affect the edge selection. 
# To prevent dublicating edges in these cases search within one order at a time.
orders = set(res.CPM_order for res in result_edges_arr)
for order in orders:
    cond = lambda res: res.atlas == 'Destrieux' and res.CPM_order == order and res.p_threshold == '0.01 p_value'
    filtered_result_edges = list(filter(cond, result_edges_arr))
    # The number of results.
    N = len(filtered_result_edges)
    print(f'The order is {order}.')
    if N != 0: 
        all_edges = set.union(*[set(res.valuable_edges) for res in filtered_result_edges])
        print(f'The number of unique edges is {len(all_edges)}.')
        for edge in all_edges:
            # The number of inlusions of the edge.
            variants = [res for res in filtered_result_edges if edge in res.valuable_edges]
            N_incl = len(variants)
            if N_incl > 1:
                print(f'The {edge} is included in {N_incl} results out of {N}.')
                for variant in variants: print(f'The variants are: \n{variant.dataset}, {variant.freq}\n')
    print('------------------------------------------------------------------')
    