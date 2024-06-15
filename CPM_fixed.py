# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 21:48:45 2023

@author: Admin
"""
import numpy as np 
from scipy import stats 
from scipy.special import btdtr
import random

def partial_corr(x, y, covar):
    xy_corr = stats.pearsonr(x, y)[0]
    xcovar_corr = stats.pearsonr(x, covar)[0]
    ycovar_corr = stats.pearsonr(y, covar)[0]
    part_corr = (xy_corr - xcovar_corr * ycovar_corr) / np.sqrt((1 - xcovar_corr ** 2) * (1 - ycovar_corr ** 2))
    n = len(x)
    k = 1 # Only one covariate is supported for now. 
    ab = (n - k) / 2 - 1
    pval = 2 * btdtr(ab, ab, 0.5 * (1 - abs(np.float64(part_corr))))
    return part_corr, pval

def poly_generator(order, fit):
    def p(x):
        sum_ = 0
        for i in range(order):
            sum_ += fit[i] * x ** (order - i)
        return sum_ + fit[order]
    return p

def train_cpm(ipmat, pheno, order=1, corr='corr', covar = None, 
              weighted = False, p_threshold=0.001):
    """
    Trains CPM with an array of conncetivity matrices and an array of 
    phenotype data.

    Parameters
    ----------
    ipmat : NxM array
        An array of M flattened connectivity matrices of size N.
    pheno : Mx1 array
        An array of M phenotype values.

    Returns
    -------
    fit_pos : 2-element array or np.nan
        Coefficients of a linear fit built with positively correlated edges;
        NaN if no fit was performed.
    fit_neg : 2-element array or np.nan
        Coefficients of a linear fit built with negatively correlated edges;
        NaN if no fit was performed.
    pe : Mx1 array
        An M sums of valueable positively correlated edges, one per subject.
    ne : Mx1 array
        An M sums of valueable negatively correlated edges, one per subject.

    """
    # start = time.time()
    if corr == 'corr':
        cc=[stats.pearsonr(pheno,im) for im in ipmat]
        rmat=np.array([c[0] for c in cc])
        pmat=np.array([c[1] for c in cc])
    elif corr == 'partial':
        cc = []
        for im in ipmat:
            cc.append(partial_corr(x = im, y = pheno, covar = covar))
        rmat=np.array([c[0] for c in cc]).squeeze()
        pmat=np.array([c[1] for c in cc]).squeeze()
    
    # stop = time.time()
    # print(stop - start)
    posedges=(rmat > 0) & (pmat < p_threshold)
    negedges=(rmat < 0) & (pmat < p_threshold)
    pe=ipmat[posedges,:]
    ne=ipmat[negedges,:]
    if weighted:
        pe_buf, ne_buf = np.zeros((pe.shape[1],)), np.zeros((ne.shape[1],))
        for i in range(pe.shape[1]):
            pe_buf[i] = np.dot(rmat[posedges], pe[:, i])
        for i in range(ne.shape[1]):
            ne_buf[i] = np.dot(np.abs(rmat[negedges]), ne[:, i])
        pe = pe_buf
        ne = ne_buf
    else:
        pe=pe.sum(axis=0)
        ne=ne.sum(axis=0)
        


    if np.sum(pe) != 0:
        fit_pos=np.polyfit(pe,pheno,order)
    else:
        fit_pos=np.nan

    if np.sum(ne) != 0:
        fit_neg=np.polyfit(ne,pheno,order)
    else:
        fit_neg=np.nan

    return fit_pos, fit_neg, pe, ne, posedges, negedges, rmat

def k_fold_validation(x, y, order=1, corr='corr', covar=None, weighted=False, 
                   p_threshold=0.001, k=10, seed=42):
    subj_count = len(y)
    behav_pred_pos = np.zeros([subj_count])
    behav_pred_neg = np.zeros([subj_count])
    all_posedges, all_negedges = [], []
    res_posedges, res_negedges = np.ones((x.shape[0],), dtype=bool), np.ones((x.shape[0],), dtype=bool)
    rand_idcs = np.arange(0, subj_count)
    random.seed(seed)
    random.shuffle(rand_idcs)
    sample_size = int(np.floor(float(subj_count) / k))
    
    for fold in range(k):
        start_idx = fold * sample_size
        end_idx = (fold + 1) * sample_size
        
        if fold != k-1:
            test_idcs = rand_idcs[start_idx:end_idx]
        else:
            test_idcs = rand_idcs[start_idx:]   
        train_idcs = rand_idcs[~np.isin(rand_idcs, test_idcs)]
        
        train_x = x[:,train_idcs]
        train_y = y[train_idcs]
        if not covar is None:
            train_covar = covar[train_idcs]
        else:
            train_covar = None
 
        test_x = x[:,test_idcs]
        
        fit_pos, fit_neg, pe, ne, posedges, negedges, rmat = \
        train_cpm(train_x, train_y, order, corr, train_covar, 
                  weighted = weighted, p_threshold=p_threshold)
        all_posedges.append(posedges)
        all_negedges.append(negedges)
        if np.any(posedges):
            res_posedges &= posedges
        if np.any(negedges):
            res_negedges &= negedges
            
        pos_poly = poly_generator(order, fit_pos)
        neg_poly = poly_generator(order, fit_neg)
        
        if weighted:
            pe_test = np.dot(rmat[posedges], test_x[posedges])
            ne_test = np.dot(np.abs(rmat[negedges]), test_x[negedges])
        else:
            pe_test = np.sum(test_x[posedges], axis=0)
            ne_test = np.sum(test_x[negedges], axis=0)
        if not np.any(np.isnan(fit_pos)):
            behav_pred_pos[test_idcs] = pos_poly(pe_test)
        else:
            behav_pred_pos[test_idcs] = np.nan
    
        if not np.any(np.isnan(fit_neg)):
           behav_pred_neg[test_idcs] = neg_poly(ne_test)
        else:
            behav_pred_neg[test_idcs] = np.nan
    
    return behav_pred_pos, behav_pred_neg, res_posedges, res_negedges, \
        all_posedges, all_negedges
    
    
def LOO_validation(x, y, order=1, corr='corr', covar=None, weighted = False, 
                   p_threshold=0.001):
    subj_count = len(y)
    behav_pred_pos = np.zeros([subj_count])
    behav_pred_neg = np.zeros([subj_count])
    # These are the graphs of valueable edges.
    all_posedges, all_negedges = [], []
    # This is the resulting graph of valueable edges obtaining from 
    # intersection of all graphs gained at every validation step.
    res_posedges, res_negedges = np.zeros((x.shape[0],)), np.zeros((x.shape[0],))
    for loo in range(0, subj_count):
        train_x = np.delete(x, [loo], axis=1)
        train_y = np.delete(y, [loo], axis=0)
        if not covar is None:
            train_covar = np.delete(covar, [loo], axis=0)
        else:
            train_covar = None
        
        test_x = x[:, loo]
        fit_pos, fit_neg, pe, ne, posedges, negedges, rmat = \
        train_cpm(train_x, train_y, order, corr, train_covar, 
                  weighted = weighted, p_threshold=p_threshold)
        if np.any(posedges):
            all_posedges.append(posedges)
            res_posedges += posedges.astype(int)
        if np.any(negedges):
            all_negedges.append(negedges)
            res_negedges += negedges.astype(int)
        
        pos_poly = poly_generator(order, fit_pos)
        neg_poly = poly_generator(order, fit_neg)
        
        if weighted:
            pe_test = np.dot(rmat[posedges], test_x[posedges])
            ne_test = np.dot(np.abs(rmat[negedges]), test_x[negedges])
        else:
            pe_test = np.sum(test_x[posedges])
            ne_test = np.sum(test_x[negedges])
        if not np.any(np.isnan(fit_pos)):
            behav_pred_pos[loo] = pos_poly(pe_test)
        else:
            behav_pred_pos[loo] = np.nan
    
        if not np.any(np.isnan(fit_neg)):
           behav_pred_neg[loo] = neg_poly(ne_test)
        else:
            behav_pred_neg[loo] = np.nan
    
    edge_percent = 0.95
    pos_edge_threshold = np.floor(edge_percent * len(all_posedges))
    neg_edge_threshold = np.floor(edge_percent * len(all_negedges))
    
    # res_posedges[res_posedges >= pos_95_threshold] = 1
    # res_posedges[res_posedges < pos_95_threshold] = 0
    # res_negedges[res_negedges >= neg_95_threshold] = 1
    # res_negedges[res_negedges < neg_95_threshold] = 0
    
    # res_posedges = res_posedges.astype(bool)
    # res_negedges = res_negedges.astype(bool)
    
    res_posedges = res_posedges >= pos_edge_threshold
    res_negedges = res_negedges >= neg_edge_threshold
    
    return behav_pred_pos, behav_pred_neg, res_posedges, res_negedges, \
        all_posedges, all_negedges
