# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 21:48:45 2023

@author: Admin
"""
import numpy as np 
from scipy import stats 
from scipy.special import betainc
import random
from sklearn.model_selection import KFold
from sklearn.linear_model import HuberRegressor
import concurrent.futures
from functools import partial
import os

def partial_corr(x, y, covar):
    xy_corr = stats.pearsonr(x, y)[0]
    xcovar_corr = stats.pearsonr(x, covar)[0]
    ycovar_corr = stats.pearsonr(y, covar)[0]
    part_corr = (xy_corr - xcovar_corr * ycovar_corr) / np.sqrt((1 - xcovar_corr ** 2) * (1 - ycovar_corr ** 2))
    n = len(x)
    k = 1 # Only one covariate is supported for now. 
    ab = (n - k) / 2 - 1
    pval = 2 * betainc(ab, ab, 0.5 * (1 - abs(np.float64(part_corr))))
    return part_corr, pval

def partial_corr2(x, y, covar1, covar2):
    xy_corr = partial_corr(x, y, covar1)[0]
    xcovar_corr = partial_corr(x, covar2, covar1)[0]
    ycovar_corr = partial_corr(y, covar2, covar1)[0]
    part_corr = (xy_corr - xcovar_corr * ycovar_corr) / np.sqrt((1 - xcovar_corr ** 2) * (1 - ycovar_corr ** 2))
    n = len(x)
    k = 2 
    ab = (n - k) / 2 - 1
    pval = 2 * betainc(ab, ab, 0.5 * (1 - abs(np.float64(part_corr))))
    return part_corr, pval

def poly_generator(order, fit):
    def p(x):
        sum_ = 0
        for i in range(order):
            sum_ += fit[i] * x ** (order - i)
        return sum_ + fit[order]
    return p

def train_cpm(ipmat, pheno, order=1, corr='corr', age = None, sex = None,
              weighted = False, p_threshold=0.001, robustRegression = True):
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
    if corr == 'corr':
        cc=[stats.pearsonr(pheno,im) for im in ipmat]
        rmat=np.array([c[0] for c in cc])
        pmat=np.array([c[1] for c in cc])
    elif corr == 'partial':
        cc = []
        for im in ipmat:
            cc.append(partial_corr(x = im, y = pheno, covar = age))
        rmat=np.array([c[0] for c in cc]).squeeze()
        pmat=np.array([c[1] for c in cc]).squeeze()
    elif corr == 'partial2':
        cc = []
        for im in ipmat:
            cc.append(partial_corr2(x = im, y = pheno, covar1 = age, covar2 = sex))
        rmat=np.array([c[0] for c in cc]).squeeze()
        pmat=np.array([c[1] for c in cc]).squeeze()
    
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
        if robustRegression:
            huber = HuberRegressor()
            huber.fit(pe.reshape(-1, 1), pheno)
            fit_pos = np.array([huber.coef_[0], huber.intercept_])
        else:
            fit_pos=np.polyfit(pe,pheno,order)
    else:
        fit_pos=np.nan

    if np.sum(ne) != 0:
        if robustRegression:
            huber = HuberRegressor()
            huber.fit(ne.reshape(-1, 1), pheno)
            fit_neg = np.array([huber.coef_[0], huber.intercept_])
        else:
            fit_neg=np.polyfit(ne,pheno,order)
    else:
        fit_neg=np.nan

    return fit_pos, fit_neg, pe, ne, posedges, negedges, rmat

def _run_one_step(train_test_idcs, x, y, age, sex, order, corr, weighted, p_threshold, robustRegression):
    train_idcs, test_idcs = train_test_idcs
    train_x = x[train_idcs].T
    train_y = y[train_idcs]
    if not age is None:
        train_age = age[train_idcs]
    if not sex is None:
        train_sex = sex[train_idcs]
    else:
        train_sex = None
    
    test_x = x[test_idcs].T
    
    fit_pos, fit_neg, pe, ne, posedges, negedges, rmat = \
    train_cpm(train_x, train_y, order, corr, train_age, train_sex,
              weighted = weighted, p_threshold = p_threshold, 
              robustRegression = robustRegression)
    if weighted:
        pe_test = np.dot(rmat[posedges], test_x[posedges])
        ne_test = np.dot(np.abs(rmat[negedges]), test_x[negedges])
    else:
        pe_test = np.sum(test_x[posedges], axis=0)
        ne_test = np.sum(test_x[negedges], axis=0)
    
    pos_poly = poly_generator(order, fit_pos)
    neg_poly = poly_generator(order, fit_neg)
    
    if not np.any(np.isnan(fit_pos)):
        behav_pred_pos = pos_poly(pe_test)
    else:
        behav_pred_pos = np.nan

    if not np.any(np.isnan(fit_neg)):
       behav_pred_neg = neg_poly(ne_test)
    else:
        behav_pred_neg = np.nan
    
    return test_idcs, behav_pred_pos, behav_pred_neg, posedges, negedges
        
def KFold_validation(x, y, order=1, corr='corr', age = None, sex = None, weighted=False, 
                   p_threshold=0.001, robustRegression = True, k=10):
    subj_count = len(y)
    cv = KFold(n_splits = k, shuffle = False)
    all_posedges, all_negedges = [None] * len(y), [None] * len(y)
    res_posedges, res_negedges = np.ones((x.shape[1],), dtype=bool), np.ones((x.shape[1],), dtype=bool)
    behav_pred_pos = np.zeros([subj_count])
    behav_pred_neg = np.zeros([subj_count])
    partial_run_one_step = partial(_run_one_step, x = x, y = y, age = age, 
                                   sex = sex, order = order, corr = corr, 
                                   weighted = weighted, p_threshold = p_threshold,
                                   robustRegression = robustRegression)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(partial_run_one_step, cv.split(x, y))
        for result in results:
            test_idcs = result[0]
            pred_pos = result[1]
            pred_neg = result[2]
            posedges = result[3]
            negedges = result[4]
            behav_pred_pos[test_idcs] = pred_pos
            behav_pred_neg[test_idcs] = pred_neg
            for idx in test_idcs:
                all_posedges[idx] = posedges
                all_negedges[idx] = negedges
            if np.any(posedges):
                res_posedges &= posedges
            if np.any(negedges):
                res_negedges &= negedges
    
    return behav_pred_pos, behav_pred_neg, res_posedges, res_negedges, \
        all_posedges, all_negedges
    
# This realization was not changed to the multithreaded version and thus is no longer supported. 
# You can uncomment and use it if you want, but you'd better use the KFold variant with 
# k = len(your_data) as this is literally the same as LOO.

# def LOO_validation(x, y, order=1, corr='corr', age=None, sex=None, weighted = False, 
#                    p_threshold=0.001, robustRegression = True):
#     subj_count = len(y)
#     behav_pred_pos = np.zeros([subj_count])
#     behav_pred_neg = np.zeros([subj_count])
#     # These are the graphs of valueable edges.
#     all_posedges, all_negedges = [], []
#     # This is the resulting graph of valueable edges obtaining from 
#     # intersection of all graphs gained at every validation step.
#     res_posedges, res_negedges = np.zeros((x.shape[0],)), np.zeros((x.shape[0],))
#     for loo in range(0, subj_count):
#         train_x = np.delete(x, [loo], axis=1)
#         train_y = np.delete(y, [loo], axis=0)
#         if not age is None:
#             train_age = np.delete(age, [loo], axis=0)
#         else:
#             train_age = None
#         if not sex is None:
#             train_sex = np.delete(sex, [loo], axis=0)
#         else:
#             train_sex = None
        
#         test_x = x[:, loo]
#         fit_pos, fit_neg, pe, ne, posedges, negedges, rmat = \
#         train_cpm(train_x, train_y, order, corr, train_age, train_sex, 
#                   weighted = weighted, p_threshold = p_threshold, 
#                   robustRegression = robustRegression)
#         if np.any(posedges):
#             all_posedges.append(posedges)
#             res_posedges += posedges.astype(int)
#         if np.any(negedges):
#             all_negedges.append(negedges)
#             res_negedges += negedges.astype(int)
        
#         pos_poly = poly_generator(order, fit_pos)
#         neg_poly = poly_generator(order, fit_neg)
        
#         if weighted:
#             pe_test = np.dot(rmat[posedges], test_x[posedges])
#             ne_test = np.dot(np.abs(rmat[negedges]), test_x[negedges])
#         else:
#             pe_test = np.sum(test_x[posedges])
#             ne_test = np.sum(test_x[negedges])
#         if not np.any(np.isnan(fit_pos)):
#             behav_pred_pos[loo] = pos_poly(pe_test)
#         else:
#             behav_pred_pos[loo] = np.nan
    
#         if not np.any(np.isnan(fit_neg)):
#            behav_pred_neg[loo] = neg_poly(ne_test)
#         else:
#             behav_pred_neg[loo] = np.nan
    
#     edge_percent = 0.95
#     pos_edge_threshold = np.floor(edge_percent * len(all_posedges))
#     neg_edge_threshold = np.floor(edge_percent * len(all_negedges))
    
#     res_posedges = res_posedges >= pos_edge_threshold
#     res_negedges = res_negedges >= neg_edge_threshold
    
#     return behav_pred_pos, behav_pred_neg, res_posedges, res_negedges, \
#         all_posedges, all_negedges
