# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 15:11:17 2024

@author: Admin
"""

# This script compares the samples in terms of demography. 

import intel_utils
import glob 
import numpy as np
import scipy as sp
import statsmodels.api as sm
import scikit_posthocs

def get_sex_count(sex_by_sample, sample):
    return [len(sex_by_sample[sample][sex_by_sample[sample] == 'M']), len(sex_by_sample[sample][sex_by_sample[sample] == 'F'])]

# Method and altas are irrelevant for this analysis (age and sex of participants
# obviously do not depend on them) and remained here only for the sake simplicity
# (I just copypasted it from CPM_base script and was too lazy to change paths
# these parameters are included in). 
method = 'wPLI' # PLV, wPLI or imCoh
atlas = 'Destrieux' # Destrieux or DK

samples = ['Chel', 'Cuba', 'German']
ages_by_sample = {'Chel': [], 'Cuba': [], 'German': []}
intels_by_sample = {'Chel': [], 'Cuba': [], 'German': []}
sex_by_sample = {'Chel': [], 'Cuba': [], 'German': []}

for sample in samples:
    if sample != 'German': 
        intel_test = '' 
    else: 
        intel_test = 'LPS'
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
    
    behav = intel_method()
    age = age_method()
    sex = sex_method()
    
    conn_dir = f'Matrices\\{method}\\{atlas}\\{sample}\\'
    files = glob.glob(conn_dir + '*.npy')
    
    label_dir = f'Labels\\{atlas}\\'
    matrices_list = []
    complete_intel_arr = []
    complete_age_arr = []
    complete_sex_arr = []
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
        complete_sex_arr.append(sex[subj])
        subj_arr.append(subj)
        # print(sex[subj])
    
    matrices = np.vstack(matrices_list)
    complete_intel_arr = np.array(complete_intel_arr)
    complete_age_arr = np.array(complete_age_arr)
    complete_sex_arr = np.array(complete_sex_arr)
    
    ages_by_sample[sample] = complete_age_arr
    intels_by_sample[sample] = complete_intel_arr
    sex_by_sample[sample] = complete_sex_arr
    
    intel_mean = np.mean(complete_intel_arr)
    intel_std = np.std(complete_intel_arr)
    intel_norm_res = sp.stats.kstest((complete_intel_arr - intel_mean) / intel_std, sp.stats.norm.cdf)
    age_mean = np.mean(complete_age_arr)
    age_std = np.std(complete_age_arr)
    age_norm_res = sp.stats.kstest((complete_age_arr - age_mean) / age_std, sp.stats.norm.cdf)
    print(f'{sample} intel \n {intel_norm_res}')
    print(f'{sample} age \n {age_norm_res}')
    

norm_chel = (intels_by_sample['Chel'] - np.min(intels_by_sample['Chel'])) / (np.max(intels_by_sample['Chel']) - np.min(intels_by_sample['Chel']))
norm_cuba = (intels_by_sample['Cuba'] - np.min(intels_by_sample['Cuba'])) / (np.max(intels_by_sample['Cuba']) - np.min(intels_by_sample['Cuba']))
norm_german = (intels_by_sample['German'] - np.min(intels_by_sample['German'])) / (np.max(intels_by_sample['German']) - np.min(intels_by_sample['German']))
intel_stats_res = sp.stats.kruskal(norm_chel, norm_cuba, norm_german)
posthoc_intel_res = scikit_posthocs.posthoc_dunn([norm_chel, norm_cuba, norm_german], p_adjust='holm')
age_stat_res = sp.stats.kruskal(ages_by_sample['Chel'], ages_by_sample['Cuba'], ages_by_sample['German'])
posthoc_age_res = scikit_posthocs.posthoc_dunn([ages_by_sample['Chel'], ages_by_sample['Cuba'], ages_by_sample['German']], p_adjust='holm')
men_count = np.array([get_sex_count(sex_by_sample, 'Chel')[0], 
             get_sex_count(sex_by_sample, 'Cuba')[0],
             get_sex_count(sex_by_sample, 'German')[0]])
total_count = np.array([len(sex_by_sample['Chel']), 
               len(sex_by_sample['Cuba']),
               len(sex_by_sample['German'])])
sex_diff_res = sm.stats.proportions_chisquare_allpairs(men_count, total_count)