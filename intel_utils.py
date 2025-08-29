# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 18:23:57 2022

@author: Admin
"""

import csv
import numpy as np
import scipy as sp
import mne
import glob
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass

fmin_arr = (4, 4,  8,  8,  10, 13, 20, 30, 30, 4)
fmax_arr = (8, 30, 13, 10, 13, 20, 30, 40, 45, 45)
# A weirdo for the sake of usability. 
freq_idcs = {(fmin, fmax): i for i, (fmin, fmax) in enumerate(zip(fmin_arr, fmax_arr))}

@dataclass
class Result:
   p_threshold: str
   CPM_order: int
   corr_mode: str
   atlas: str
   dataset: str
   method: str
   corr_sign: str
   freq: str
   r: float
   p: float
   corrected_p: float
   r2: float
   behav: np.ndarray
   pred: np.ndarray
   mae: float
   rmse: float

   def __str__(self):
       return f'p-value threshold: {self.p_threshold}, CPM order: {self.CPM_order}, atlas: {self.atlas}, dataset: {self.dataset}\n' + \
   f'FC method: {self.method}, correlation sign: {self.corr_sign}, frequency: {self.freq}, r: {round(self.r, 3)}, p: {round(self.p, 8)}, corrected p: {round(self.corrected_p, 4)}'

@dataclass
class SteigerResult:
   p_threshold: str
   CPM_order: int
   corr_mode: str
   atlas: str
   dataset: str
   method: str
   corr_sign: str
   freq: str
   excluded_edge: str
   pearson_r: float
   pearson_p: float
   corrected_pearson_p: float
   steiger_z: float
   steiger_p: float
   corrected_steiger_p: float
                
@dataclass 
class ResultEdges:
    p_threshold: str
    CPM_order: int
    corr_mode: str
    atlas: str
    dataset: str
    method: str
    corr_sign: str
    freq: str
    valuable_edges: list[str]

    
def steiger_res_from_res(result):
    steiger_res = SteigerResult(p_threshold=result.p_threshold, CPM_order=result.CPM_order,
                                corr_mode=result.corr_mode, atlas=result.atlas, dataset=result.dataset,
                                method=result.method, corr_sign=result.corr_sign,
                                freq=result.freq, excluded_edge='', pearson_r=np.nan, 
                                pearson_p=np.nan, corrected_pearson_p=np.nan, steiger_z=np.nan,
                                steiger_p=np.nan, corrected_steiger_p=np.nan)
    return steiger_res

def parse_result_file(res_dir, der_res_dir):
    raw_results = read_csv(der_res_dir + '\\results.csv')[1:]
    results = []
    for result in raw_results:
        p_threshold = result[0]
        CPM_order = result[1]
        corr_mode = 'partial'
        atlas = result[2]
        dataset = result[3]
        method = result[4] 
        corr_sign = result[5]
        freq = result[6]
        r = float(result[7])
        p = float(result[8])
        corrected_p = float(result[9])
        r2 = float(result[10])
        mae = float(result[11])
        rmse = float(result[12])
        
        behav = np.load(f'{res_dir}\\{method}\\{atlas} {dataset}\\{corr_mode}\\{CPM_order}\\{p_threshold}\\memory_{freq}.npy')
        pred = np.load(f'{res_dir}\\{method}\\{atlas} {dataset}\\{corr_mode}\\{CPM_order}\\{p_threshold}\\{corr_sign}_pred_{freq}.npy')
        result = Result(p_threshold = p_threshold, CPM_order = CPM_order, corr_mode = corr_mode, 
                        atlas = atlas, dataset = dataset, method = method, 
                        corr_sign = corr_sign, freq = freq, r = r, 
                        p = p, corrected_p = corrected_p, r2 = r2, 
                        behav = behav, pred = pred, mae = mae, rmse = rmse)
        results.append(result)
    return results
   
def modularity_curried(community_method):
    def func(G, weight):
        return nx.community.modularity(G, community_method(G), weight)
    return func
topology_methods = {'Shortest path length': nx.algorithms.average_shortest_path_length,
           'Clustering': nx.algorithms.average_clustering,
           'Modularity': modularity_curried(nx.algorithms.community.louvain_communities)}
topology_method_names = topology_methods.keys()

def read_csv(file):
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile)
        arr = np.array([line for line in reader])
    return arr

def get_topology(files, freq_of_interest, 
                 methods=topology_method_names):
    metrics_values = np.zeros((len(files), len(freq_of_interest), len(methods)))
    for i, file in enumerate(files):
        conn = np.load(file)
        for j, freq in enumerate(freq_of_interest):
            conn_freq = conn[j][conn[j] != 0]
            conn_matrix = sp.spatial.distance.squareform(conn_freq)
            G = nx.from_numpy_array(conn_matrix, create_using=nx.Graph)
            for k, method_name in enumerate(methods):
                    metric = topology_methods[method_name](G, weight='weight')
                    metrics_values[i, j, k] = metric
    return metrics_values
    
def get_Cuban_topology(freq_of_interest, conn_method,
                       topology_methods=topology_method_names):
    """
    Computes the topology metrics for the conncetivity graphs using the Cuban
    sample in frequency bands defined by freq_of_interest. 

    Parameters
    ----------
    freq_of_interest : Nx1 array
        An array of frequency band indices. Can be defined either by extracting
        indices from fmin_arr and fmax_arr manually or by using freq_idcs dictionary.
    conn_method : string
        A connectivity method to use. Available options are 'wPLI', 'PLV' and 'ciPLV'.
    topology_methods : Kx1 array of strings, optional
        Names of topology metrics to calculate. Available options are 
        'Shortest path length', 'Clustering' and 'Modularity'. The default is 
        topology_method_names, that is, all of these methods.

    Returns
    -------
    MxNxK array
        An array of values of topology metrics for M participants, N frequency
        bands and K topology metrics.

    """
    data_dir = f'E:\Work\MBS\EEG\Modelling\Intelligence\Cuban Map Project\Matrices\{conn_method}'
    files = glob.glob(data_dir + '\\CBM*.npy')
    return get_topology(files, freq_of_interest, topology_method_names)

def get_spectrum(files, fmin_arr=fmin_arr, fmax_arr=fmax_arr, picked_chs=['eeg']):
    psd_arr = np.zeros((len(files), len(fmin_arr)))
    for i, file in enumerate(files):
        epochs = mne.read_epochs(file)
        epoch_by_chs = epochs.pick(picked_chs)
        for j, (fmin, fmax) in enumerate(zip(fmin_arr, fmax_arr)):
            psd = epoch_by_chs.compute_psd(fmin=fmin, fmax=fmax, n_jobs=-1)
            psd = np.mean(psd, axis=(0, 1, 2))
            psd_arr[i, j] = psd
    return psd_arr
    
def get_Cuban_spectrum(fmin_arr=fmin_arr, fmax_arr=fmax_arr, picked_chs=['eeg']):
    """
    Computes the power spectral density using the data of the Cuban sample in 
    frequency bands defined by fmin_arr and fmax_arr. 

    Parameters
    ----------
    fmin_arr : Nx1 array, optional
        An array of lower bounds of frequency bands. The default is fmin_arr.
    fmax_arr : Nx1 array, optional
        An array of upper bounds of frequency bands. The default is fmax_arr.
    picked_chs : array of strings, optional
        Channels to compute PSD onto. Can be names or types of channels. If set
        types, all channels of the chosen type are picked. The default is ['eeg'].

    Returns
    -------
    MxN array
        An array of PSD values for M participants in N frequency bands. The result
        is averaged over epochs, channels and frequency points. 

    """
    data_dir = 'E:\Work\MBS\EEG\Modelling\Intelligence\Cuban Map Project\Preprocessed data'
    files = glob.glob(data_dir + '\\CBM*.fif')
    return get_spectrum(files, fmin_arr, fmax_arr, picked_chs)

def Raven_to_IQ(raven_arr, raven_IQ_map):
    raven_values = np.array([0 if el == 'NA' else int(el) for el in raven_arr])
    raven_sum = np.sum(raven_values)
    if raven_sum < 15 or raven_sum > 60: return np.nan
    return raven_IQ_map[raven_sum]

def get_Raven_intel():
    """
    Gets Raven intelligence score from the Ivanov's (or Zakharov's, God knows)
    file df_ML_full.csv. Assumes there exist Raven data\\RavenIQ.csv and
    Raven data\\df_ML_full.csv directories, the first one contains a rule on
    how to map Raven's score onto IQ values, the second one contains Raven's
    scores themselves. Takes only long Raven results. Strictly dataset- and
    project-specific, so don't expect any generalization - the only reason why
    I extracted this was that I was fed up with copy-pasting the same lines of
    code from script to script. 

    Returns
    -------
    raven_IQ_dict : str->int dict
        Returns a dictionary that maps participant IDs onto their IQ scores. 
        Can contain NaN if raw Raven scores sum up to unbelievably low (< 15) 
        or impossibly high value (> 60). 

    """
    # Form a rule on how to map Raven scores to IQ.
    
    raven_IQ = read_csv('Raven data\\RavenIQ.csv')
    raven_IQ_map = {row[0].astype(int): row[1].astype(int) for row in raven_IQ.T}
    
    # The first row is a header, so is omitted. The columns 0 through 61 are IDs
    # and raw Raven results. The short Raven (without first sections) is excluded. 
    
    raw_behav_raven = read_csv('Raven data\\df_ML_full.csv')[1:,0:61]
    raw_behav_raven = raw_behav_raven[raw_behav_raven[:, 1] != 'NA']
    
    # Sum up the raw Raven scores and build the resulting dict.
    
    raven_IQ_dict = {row[0]: Raven_to_IQ(row[1:], raven_IQ_map) for row in raw_behav_raven}
    return raven_IQ_dict

def get_Raven_sex():
    raw_sex_raven = read_csv('Raven data\\gender.csv')
    def sex_selector(x): 
        if x == '2': return 'F'
        if x == '1': return 'M'
        return 'N/A'
    sex_dict = {row[1]: sex_selector(row[0]) for row in raw_sex_raven}
    return sex_dict
    
def get_Raven_age():
    raw_behav_raven = read_csv('Raven data\\df_ML_full.csv')[1:,[0,79]]
    # raw_behav_raven = raw_behav_raven[raw_behav_raven[:, 1] != 'NA']
    raven_age_dict = {row[0]: int(row[1]) if row[1] != 'NA' and row[1] != '0' else np.nan 
                      for row in raw_behav_raven}
    return raven_age_dict

def get_WAIS_intel(return_test='PIQ'):
    """
    Gets WAIS intelligence score from Cuban Brain Map Project, namely, 
    WAIS_III.csv file, assuming the file exists in the same directory the 
    intel_utils.py (this script file) does. Strictly dataset- and
    project-specific, so don't expect any generalization - the only reason why
    I extracted this was that I was fed up with copy-pasting the same lines of
    code from script to script. 

    Parameters
    ----------
    return_PIQ : bool, optional
        If True, returns PIQ (pefrofmance IQ) values, FSIQ (full-scale IQ)
        otherwise. The default is True.

    Returns
    -------
    PIQ_dict : str->int dict
        Returns a dictionary that maps participant IDs onto their IQ scores.

    """
    if return_test == 'PIQ':
        IQ_index = 3  
    elif return_test == 'FSIQ':
        IQ_index = 1
    elif return_test == 'VIQ':
        IQ_index = 2
    behav_PIQ = read_csv('WAIS_III.csv')
    
    # Two first rows are title and header, so are omitted. 0th column is 
    # participant ID, 3rd one is PIQ, 1st one is FSIQ.
    
    PIQ_with_ID = behav_PIQ[2:, [0, IQ_index]]
    
    # For some participants there's no any intelligence data, so remove them.
    
    PIQ_with_ID_space_removed = PIQ_with_ID[PIQ_with_ID[:, 1] != '']
    PIQ_dict = {row[0]: row[1].astype(int) for row in PIQ_with_ID_space_removed}
    return PIQ_dict

def get_WAIS_sex():
    data = read_csv('Demographic_data.csv')   
    sex = data[2:, [0, 1]]
    sex_dict = {row[0]: row[1] for row in sex}
    return sex_dict

def get_WAIS_age():
    data = read_csv('Demographic_data.csv')   
    age = data[2:, [0, 2]]
    age_dict = {row[0]: row[1].astype(int) for row in age}
    return age_dict
  
def get_LPS_intel():
    """
    Gets LPS (Leistungsprüfsystem) intelligence score from Leipzig Mind-Brain-Body 
    project, namely, LPS.csv.csv file. Strictly dataset- and project-specific, 
    so don't expect any generalization - the only reason why I extracted this was 
    that I was fed up with copy-pasting the same lines of code from script to script.
    I'd rather copy-paste this description instead. 

    Returns
    -------
    wst_dict : str->int dict
        Returns a dictionary that maps participant IDs onto their IQ scores.

    """
    intel = read_csv('F:\EEG_data\German dataset\MPI-Leipzig_Mind-Brain-Body-LEMON\Behavioural_Data_MPILMBB_LEMON\Cognitive_Test_Battery_LEMON\LPS\\LPS.csv')
    lps = intel[1:]
    lps_dict = {row[0]: int(row[1]) if row[2] == '' else np.nan for row in lps}
    return lps_dict

def get_WST_intel():
    intel = read_csv('F:\EEG_data\German dataset\MPI-Leipzig_Mind-Brain-Body-LEMON\Behavioural_Data_MPILMBB_LEMON\Cognitive_Test_Battery_LEMON\WST\\WST.csv')
    wst = intel[1:, [0, 3]]
    wst_dict = {row[0]: int(row[1]) for row in wst}
    return wst_dict

def get_RWT_intel():
    intel = read_csv('F:\EEG_data\German dataset\MPI-Leipzig_Mind-Brain-Body-LEMON\Behavioural_Data_MPILMBB_LEMON\Cognitive_Test_Battery_LEMON\RWT\RWT.csv')
    rwt = intel[1:]
    rwt_dict = {row[0]: float(row[1]) - float(row[3]) if row[1] != '' and row[12] == '' else np.nan for row in rwt}
    return rwt_dict
    
def get_German_sex():
    demo = read_csv('F:\EEG_data\German dataset\MPI-Leipzig_Mind-Brain-Body-LEMON\Behavioural_Data_MPILMBB_LEMON\\META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')
    sex = demo[1:, [0, 1]]
    def sex_selector(x): 
        if x == '1': return 'F'
        if x == '2': return 'M'
        return 'N/A'
    sex_dict = {row[0]: sex_selector(row[1]) for row in sex}
    return sex_dict

def get_German_age():
    demo = read_csv('F:\EEG_data\German dataset\MPI-Leipzig_Mind-Brain-Body-LEMON\Behavioural_Data_MPILMBB_LEMON\\META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv')
    age = demo[1:, [0, 2]]
    age_dict = {row[0]: (int(row[1].split('-')[1]) + int(row[1].split('-')[0])) / 2
                for row in age}
    return age_dict
    
def plot_scatter_against_function(x_data, y_data, functions, axis_labels=None,
                                  legend=None, k_lim=0.1, scatter_colours=None,
                                  curve_colours=None, title=None, ax = None):
    """
    Plots a scatter plot defined by x_data and y_data and a regular plot defined by
    functions. Typical use-case is a plotting least-squared fit against data it is
    fitted on. Allows to place several datasets and functions on the same plot,
    assuming shape consistency is satisfied.

    Parameters
    ----------
    x_data : MxN array
        A set of M datasets containing x-data and consisting of N points.
    y_data : MxN array
        A set of M datasets containing y-data and consisting of N points.
    functions : Px1 array of functions
        A set of P functions of type R->R. Must be broadcastable.
    axis_labels : 2x1 array of strings, optional
        A set of 2 string representing the axis labels. If None, no label is attached
        to the plot. The default is None.
    legend : (M+P)x1 array of strings, optional
        A set of M+P strings representing the plot legend. If None, no legend
        is attached to the plot. The default is None.
    k_lim : float, optional
        A spacing coefficient. If 0, the left and right border of the plot is
        built by the leftmost and rightmost points of x_data. The default is 0.1.
    scatter_colours : Mx1 array of strings, optional
        The colours of dots built by x_data and y_data. If None, the colours are chosen
        by default. The default is None.
    curve_colours : Px1 array of strings, optional
        The colours of curves built by functions. If None, the colours are chosen
        by default. The default is None.
    title : str, optional
        The title of the plot.
    Returns
    -------
    None.

    """
    if ax == None:
        fig = plt.figure()
        ax = fig.add_axes((0, 0, 1, 1))
    x_arr = np.array([])
    y_arr = np.array([])
    # Create flattened array out of all data arrays in order to calculate the limits
    for data in zip(x_data, y_data):
        x_arr = np.append(x_arr, data[0])
        y_arr = np.append(y_arr, data[1])

    x_min, x_max = np.min(x_arr), np.max(x_arr)
    y_min, y_max = np.min(y_arr), np.max(y_arr)

    x_lim_range = x_max - x_min
    y_lim_range = y_max - y_min
    xlim = [x_min - k_lim * x_lim_range,
            x_max + k_lim * x_lim_range] if x_lim_range > 0 else [-1, 1]
    ylim = [y_min - k_lim * y_lim_range,
            y_max + k_lim * y_lim_range] if y_lim_range > 0 else [-1, 1]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='both', labelsize = 40)
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')

    for i, (x,y) in enumerate(zip(x_data, y_data)):
        if scatter_colours != None:
            ax.scatter(x, y, c = scatter_colours[i])
        else:
            ax.scatter(x, y)

    N = 100
    x = np.linspace(x_min, x_max, N)
    for i, f in enumerate(functions):
        if curve_colours != None:
            ax.plot(x, f(x), c = curve_colours[i])
        else:
            ax.plot(x, f(x))

    if axis_labels != None:
        ax.set_xlabel(axis_labels[0], color='w', fontsize=40)
        ax.set_ylabel(axis_labels[1], color='w', fontsize=40)
    if legend != None: ax.legend(legend)
    if title != None: ax.set_title(title, color = 'w', fontsize=40)
    ax.grid()
    return ax
