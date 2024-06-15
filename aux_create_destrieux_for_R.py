# -*- coding: utf-8 -*-
"""
Created on Wed May 29 17:26:41 2024

@author: Admin
"""

# This script creates a description of a Destrieux atlas to create glass grain
# pictures in R.

import numpy as np
import mne
import csv

atlas = 'Destrieux'

labels = label_dir = f'Labels\{atlas}'
labels = np.load(f'{label_dir}\\CBM00001_labels.npy', allow_pickle=True)

lobes = ['Insula', 'Cingulate', 'Frontal', 'Occipital', 'Parietal', 'Temporal']
lobes_dict = {lobe: open(f'Destrieux lobes\\{lobe.upper()}.txt').read().split('\n') for lobe in lobes}

with open('Destrieux atlas.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    header = ['ROI.Name', 'x.mni', 'y.mni', 'z.mni', 'network', 'hemi']
    writer.writerow(header)
    for label in labels:
        if label.name.startswith('Unknown'): continue
        name = label.name
        center_of_mass = label.center_of_mass()
        hemi = 'L' if name.endswith('lh') else 'R'
        coords = np.round(mne.vertex_to_mni(center_of_mass, subject = 'fsaverage', hemis = 0 if hemi == 'L' else 1)).astype(int)
        for lobe in lobes_dict:
            if name in lobes_dict[lobe]: 
                network = lobe
                break
        writer.writerow([name, coords[0], coords[1], coords[2], network, hemi])