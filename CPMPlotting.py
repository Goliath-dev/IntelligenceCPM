# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 20:44:22 2023

@author: Admin
"""

import numpy as np
import scipy as sp
from sklearn.metrics import r2_score, mean_absolute_error
from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout
import matplotlib.pyplot as plt
import intel_utils
import os
from docx import Document


class CPMTable:
    
    def __init__(self):
        self._setup_table()
        
    def _setup_table(self):
        document = Document()
        table = document.add_table(cols = 1, rows = 5)
        side_cells = table.columns[0].cells
        side_cells[1].text = 'pos R²'
        side_cells[2].text = 'pos MAE'
        side_cells[3].text = 'neg R²'
        side_cells[4].text = 'neg MAE'
        
        self.document = document
        self.table = table
     
    def _update_table(self, r2, mae, is_pos):
        cells = self.current_cells
        idcs = (1, 2) if is_pos else (3, 4)
        cells[idcs[0]].text = str(r2)
        cells[idcs[1]].text = str(mae)
    
    def _calculate_r2_mae(self, intel_arr, correct_pred):
        r2 = round(r2_score(intel_arr, correct_pred), 2)
        mae = round(mean_absolute_error(intel_arr, correct_pred), 2)
        return (r2, mae)
      
    def add_col(self, min_freq, max_freq):
        self.current_cells = self.table.add_column(width=4).cells
        for cell in self.current_cells:
            cell.text = '-'
        self.current_cells[0].text = f'{min_freq}-{max_freq} Hz'
        
    def update_pos_table(self, intel_arr, correct_pos_pred):
        r2, mae = self._calculate_r2_mae(intel_arr, correct_pos_pred)
        self._update_table(r2, mae, is_pos = True)
    
    def update_neg_table(self, intel_arr, correct_neg_pred):
        r2, mae = self._calculate_r2_mae(intel_arr, correct_neg_pred)
        self._update_table(r2, mae, is_pos = False)
    
    def save_table(self, table_dir):
        self.document.save(table_dir + 'table.docx')
        
    
class CPMPlot:

    def __init__(self, ROI_labels, fmin, fmax):
        self._setup_figure()
        self._setup_conn_circle(ROI_labels, fmin, fmax)
    
    def _setup_figure(self):
        fig = plt.figure(figsize = (40, 40))
        fig.set_facecolor('k')
        grid_spec = fig.add_gridspec(2, 2)
        
        self.fig = fig
        self.grid_spec = grid_spec

        
    def _setup_conn_circle(self, ROI_labels, fmin, fmax):
        self.ROI_labels = ROI_labels
        self.fmin = fmin
        self.fmax = fmax
        N = len(ROI_labels)
        self.pos_matrix = np.zeros((int(N * (N - 1) / 2),))
        self.neg_matrix = np.zeros((int(N * (N - 1) / 2),))
        
        ROI_names = [label.name for label in ROI_labels]
        self.ROI_names = ROI_names

        lh_labels = [name for name in ROI_names if name.endswith('lh')]
        label_ypos = list()
        for name in lh_labels:
            idx = ROI_names.index(name)
            ypos = np.mean(ROI_labels[idx].pos[:, 1])
            label_ypos.append(ypos)
        lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]
        rh_labels = [label[:-2] + 'rh' for label in lh_labels]
        node_order = list()
        node_order.extend(lh_labels[::-1])  # reverse the order
        node_order.extend(rh_labels)  
            
        node_angles = circular_layout(ROI_names, node_order, start_pos=90,
                                  group_boundaries=[0, len(ROI_names) / 2])
        self.node_angles = node_angles
        
        pos_title = f'[{fmin}-{fmax}] Hz, positive edges'
        neg_title = f'[{fmin}-{fmax}] Hz, negative edges'
        
        ax_pos = self.fig.add_subplot(self.grid_spec[0, 0], projection='polar')
        ax_pos.set_title(pos_title, color='w', y = 1.1, fontsize = 25)  
        ax_neg = self.fig.add_subplot(self.grid_spec[0, 1], projection='polar')
        ax_neg.set_title(neg_title, color='w', y = 1.1, fontsize = 25)  
        
        self.ax_pos = ax_pos
        self.ax_neg = ax_neg
        
    def _setup_scatter(self, intel_arr, correct_pred, is_pos):
        grid_position = self.grid_spec[1, 0] if is_pos else self.grid_spec[1, 1]
        ax_scatter = self.fig.add_subplot(grid_position)
        r2 = round(r2_score(intel_arr, correct_pred), 2)
        mae = round(mean_absolute_error(intel_arr, correct_pred), 2)
        title_word = 'positively' if is_pos else 'negatively'
        title = f'[{self.fmin} - {self.fmax}] Hz, R² = {r2}, MAE = {mae}, {title_word} correlated edges'
        axis_labels = ['Observed PIQ score', 'Predicted PIQ score']
        self.axis_labels = axis_labels
        
        # self.ax_pos_scatter = None
        # self.scatter_pos_title = None
        # self.intel_pos_arr = None
        # self.pos_pred = None
        # self.ax_neg_scatter = None
        # self.scatter_neg_title = None
        # self.intel_neg_arr = None
        # self.neg_pred = None
        if is_pos: 
            self.ax_pos_scatter = ax_scatter
            self.scatter_pos_title = title
            self.intel_pos_arr = intel_arr
            self.pos_pred = correct_pred
        else:
            self.ax_neg_scatter = ax_scatter
            self.scatter_neg_title = title
            self.intel_neg_arr = intel_arr
            self.neg_pred = correct_pred

    def setup_pos_scatter(self, intel_arr, correct_pos_pred):
        self._setup_scatter(intel_arr, correct_pos_pred, is_pos = True)
    
    def setup_neg_scatter(self, intel_arr, correct_neg_pred):
        self._setup_scatter(intel_arr, correct_neg_pred, is_pos = False)
     
    def set_pos_matrix(self, pos_matrix):
        self.pos_matrix = pos_matrix

    def set_neg_matrix(self, neg_matrix):
        self.neg_matrix = neg_matrix
        
    def plot_quad_plot(self, img_dir):
        plot_connectivity_circle(sp.spatial.distance.squareform(self.pos_matrix.astype(float)), 
                                 self.ROI_names, node_angles=self.node_angles,
                                 ax = self.ax_pos, fontsize_title=20, fontsize_names=20)
        plot_connectivity_circle(sp.spatial.distance.squareform(self.neg_matrix.astype(float)), 
                                 self.ROI_names, node_angles=self.node_angles,
                                 ax = self.ax_neg, fontsize_title=20, fontsize_names=20)
        # print(self.ax_pos_scatter)
        if hasattr(self, 'ax_pos_scatter'):
            intel_utils.plot_scatter_against_function([self.intel_pos_arr], 
                                                      [self.pos_pred], 
                                                      [],
                                                      axis_labels=self.axis_labels, 
                                                      title=self.scatter_pos_title, 
                                                      ax = self.ax_pos_scatter)
        if hasattr(self, 'ax_neg_scatter'):
            intel_utils.plot_scatter_against_function([self.intel_neg_arr], 
                                                      [self.neg_pred], 
                                                      [],
                                                      axis_labels=self.axis_labels, 
                                                      title=self.scatter_neg_title, 
                                                      ax = self.ax_neg_scatter)
        os.makedirs(img_dir, exist_ok=True)
        self.fig.savefig(img_dir + f'[{self.fmin}-{self.fmax}] Hz', facecolor='black')
        
        
        
        
        
        
        
        
        
        
        