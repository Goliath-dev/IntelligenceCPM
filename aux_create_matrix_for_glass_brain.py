# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:12:55 2024

@author: Admin
"""

# This script creates a matrix of p-values in lesioning approach. 

import numpy as np
import glob
import csv
import netplotbrain
# import matplotlib.pyplot as plt
import templateflow.api as tf
import pandas as pd

method = 'wPLI' # PLV, wPLI or imCoh
sample = 'German' # Cuba, Chel or German
atlas = 'Destrieux' # Destrieux or DK
freq = '4-8' # 4-8, 8-13, 8-10, 10-13, 13-20, 20-30 or 30-45
prefix = 'neg' # posedges or negedges

sample_number = {'Chel': '1st', 'Cuba': '2nd', 'German': '3rd'}
res_dir = 'Results\Imgs for article\Lesion aggr\For glass brain'
label_dir = f'Labels\{atlas}'

labels = np.load(f'{label_dir}\\CBM00001_labels.npy', allow_pickle=True)

edges_f = open(f'{res_dir}\\{sample_number[sample]} sample_{method}_{freq} Hz_{prefix}_{atlas}_edges.txt')
p_f = open(f'{res_dir}\\{sample_number[sample]} sample_{method}_{freq} Hz_{prefix}_{atlas}_p_values.txt')
p_values = list(filter(lambda s: s != '', p_f.read().split('\n')))

edges = list(filter(lambda s: ''.join(s.split()) != 'to' and ''.join(s.split()) != '', edges_f.read().split('\n')))
edges = list(''.join(edge.split()) for edge in edges)
edges = np.array(edges)

edges_from = edges[0::2]
edges_to = edges[1::2]

# I read label names from a file file rather than [label.name for label in labels]
# because I use the matrix later on in an external R script that plots the glass brain.
# Therefore, the order of labels here matters (it defines the order of nodes in the
# adjacency matrix and it must fit the order of them in the R lib so that the nodes
# would be placed correctly), and I copied the nodes from R to a txt file to preserve
# the order.
# P. S.: To be fully precise, this is only true for the DK atlas, as it is built-in
# in the R library I use. The Destrieux atlas is custom and built by me (it is not built-in there),
# so technically I could just use [label.name for label in labels], as this is exactly the 
# way I built Destrieux atlas for the R script. I used a txt file instead, though, 
# for the sake of standartization (and a bit of laziness). 
label_names = open(f'{atlas}_label_names.txt').read()
label_names = label_names.split('\n')
label_names = np.array([''.join(name.split()) for name in label_names])
idcs_from = [np.argwhere(label_names == el)[0, 0] for el in edges_from]
idcs_to = [np.argwhere(label_names == el)[0, 0] for el in edges_to]

N = 148 if atlas == 'Destrieux' else 68
matrix = np.zeros((N, N))
for i, j, p in zip(idcs_from, idcs_to, p_values):
    if '*' in p:
        float_p = float(p.split('*')[0])
        weight = 1 / float_p
        matrix[i, j] = weight
        matrix[j, i] = weight

edges_f.close()
p_f.close()

with open('matrix_glass_brain.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    header = np.arange(0, N).astype(str)
    writer.writerow(header)
    for row in matrix:
        writer.writerow(row)



# atlas = {'template': 'fsaverage',
#           'atlas': 'Destrieux2009'}
# atlasinfo = tf.get(template='fsaverage', atlas='Destrieux2009')
# atlasinfo = pd.read_csv(atlasinfo[1], sep='\t')
# netplotbrain.plot(template='fsaverage',
#                   nodes={'atlas': 'Destrieux2009'},
#                   edges=matrix,
#                   view='LS',
#                   template_style='glass')
