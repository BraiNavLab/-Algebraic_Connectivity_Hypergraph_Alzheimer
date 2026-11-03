############################################################################################
# This script computes the hyperedge weights for each subject using the four methods:
#  - Algebraic connectivity
#  - Gaussian similarity kernel
#  - Mean Pearson correlation
#  - L2-norm of the regression coefficients
#
# The hyperedge weights dataframes are created for further statistical analysis and saved.
#
# The hyperedge weights will be saved in the folder __hypergraph__
############################################################################################

import pickle
import numpy as np
from Utils.Computation import compute_hyperedges_weights, compute_extract_weights_values_from_datastructure, compute_create_dataframe_for_analysis



##### Extraction of hyperedge weights
def hyperedge_weights(fmri_by_group: list, H: np.array, idx_hyperedges_to_consider: np.array, lambda_: float, fmri_all: np.array, type_regression: str = 'lasso', keep_negative: bool = False):
    '''
    Computes hyperedge weights for all subjects and creates dataframes for further analysis.\n
    Params:
        - fmri_by_group (list): list of fMRI data arrays for each group [HC for hypergraph, HC for analysis, MCI, AD]
        - H (np.array): hypergraph incidence matrix
        - idx_hyperedges_to_consider (np.array): indices of hyperedges to consider
        - lambda_ (float): optimal lambda value for regression
        - fmri_all (np.array): fMRI data of all subjects together
        - type_regression (str): type of regression to use (default: 'lasso')
        - keep_negative (bool): whether to keep negative values in the coefficients (default: False)
    '''
    
    # Compute hyperedges weights for all classes
    print('\nComputing hyperedges weights for all subjects...')
    # Dataset HC for hypergraph
    hyperedges_weights_hc_hypergraph = compute_hyperedges_weights(fmri=fmri_by_group[0], H=H, idx_hyperedges_to_consider=idx_hyperedges_to_consider, optimal_lambda_=lambda_, fmri_all=fmri_all, type_regression=type_regression, keep_negative=keep_negative)

    # Dataset HC for analysis
    hyperedges_weights_hc = compute_hyperedges_weights(fmri=fmri_by_group[1], H=H, idx_hyperedges_to_consider=idx_hyperedges_to_consider, optimal_lambda_=lambda_, fmri_all=fmri_all, type_regression=type_regression, keep_negative=keep_negative)

    # Dataset MCI
    hyperedges_weights_mci = compute_hyperedges_weights(fmri=fmri_by_group[2], H=H, idx_hyperedges_to_consider=idx_hyperedges_to_consider, optimal_lambda_=lambda_, fmri_all=fmri_all, type_regression=type_regression, keep_negative=keep_negative)

    # Dataset AD
    hyperedges_weights_ad = compute_hyperedges_weights(fmri=fmri_by_group[3], H=H, idx_hyperedges_to_consider=idx_hyperedges_to_consider, optimal_lambda_=lambda_, fmri_all=fmri_all, type_regression=type_regression, keep_negative=keep_negative)

    # Extract metrics from data structure
    list_metrics = ['algeb_conn', 'Gaussian_kernel_simm', 'mean_corr', 'coeff_l2norm']

    W_hc_hypergraph = {metric: compute_extract_weights_values_from_datastructure(dict_hyperedges_weights=hyperedges_weights_hc_hypergraph, idx_hyperedges=idx_hyperedges_to_consider, num_subjs=fmri_by_group[0].shape[0], metric_to_extract=metric) for metric in list_metrics}    # HC-hypergraph weights
    W_hc_analysis = {metric: compute_extract_weights_values_from_datastructure(dict_hyperedges_weights=hyperedges_weights_hc, idx_hyperedges=idx_hyperedges_to_consider, num_subjs=fmri_by_group[1].shape[0], metric_to_extract=metric) for metric in list_metrics}                 # HC-analysis weights
    W_mci = {metric: compute_extract_weights_values_from_datastructure(dict_hyperedges_weights=hyperedges_weights_mci, idx_hyperedges=idx_hyperedges_to_consider, num_subjs=fmri_by_group[2].shape[0], metric_to_extract=metric) for metric in list_metrics}                        # MCI weights
    W_ad = {metric: compute_extract_weights_values_from_datastructure(dict_hyperedges_weights=hyperedges_weights_ad, idx_hyperedges=idx_hyperedges_to_consider, num_subjs=fmri_by_group[3].shape[0], metric_to_extract=metric) for metric in list_metrics}                          # AD weights

    # Create dataframes for analysis
    df_W = {metric: compute_create_dataframe_for_analysis(data_to_merge=[W_hc_analysis[metric], W_mci[metric], W_ad[metric]], idx_hyperedges=idx_hyperedges_to_consider, group_labels=['HC','MCI','AD']) for metric in list_metrics}
    df_W_check = {metric: compute_create_dataframe_for_analysis(data_to_merge=[W_hc_hypergraph[metric], W_hc_analysis[metric]], idx_hyperedges=idx_hyperedges_to_consider, group_labels=['HC for hypergraph','HC for analysis']) for metric in list_metrics}

    # Save dataframes
    dataframe_data = {
        'df_algeb_conn': df_W['algeb_conn'][0],
        'df_algeb_conn_melt': df_W['algeb_conn'][1],
        'algeb_conn_np': {
            'hc_hypergraph': W_hc_hypergraph['algeb_conn'],
            'hc_analysis': W_hc_analysis['algeb_conn'],
            'mci': W_mci['algeb_conn'],
            'ad': W_ad['algeb_conn']
        },
        'df_gaus_kernel': df_W['Gaussian_kernel_simm'][0],
        'df_gaus_kernel_melt': df_W['Gaussian_kernel_simm'][1],
        'gaus_kernel_np': {
            'hc_hypergraph': W_hc_hypergraph['Gaussian_kernel_simm'],
            'hc_analysis': W_hc_analysis['Gaussian_kernel_simm'],
            'mci': W_mci['Gaussian_kernel_simm'],
            'ad': W_ad['Gaussian_kernel_simm']
        },
        'df_mean_corr': df_W['mean_corr'][0],
        'df_mean_corr_melt': df_W['mean_corr'][1],
        'mean_corr_np': {
            'hc_hypergraph': W_hc_hypergraph['mean_corr'],
            'hc_analysis': W_hc_analysis['mean_corr'],
            'mci': W_mci['mean_corr'],
            'ad': W_ad['mean_corr']
        },
        'df_coeff_l2norm': df_W['coeff_l2norm'][0],
        'df_coeff_l2norm_melt': df_W['coeff_l2norm'][1],
        'coeff_l2norm_np': {
            'hc_hypergraph': W_hc_hypergraph['coeff_l2norm'],
            'hc_analysis': W_hc_analysis['coeff_l2norm'],
            'mci': W_mci['coeff_l2norm'],
            'ad': W_ad['coeff_l2norm']
        },
        'df_algeb_conn_check': df_W_check['algeb_conn'][0],
        'df_algeb_conn_melt_check': df_W_check['algeb_conn'][1],
        'df_gaus_kernel_check': df_W_check['Gaussian_kernel_simm'][0],
        'df_gaus_kernel_melt_check': df_W_check['Gaussian_kernel_simm'][1],
        'df_mean_corr_check': df_W_check['mean_corr'][0],
        'df_mean_corr_melt_check': df_W_check['mean_corr'][1],
        'df_coeff_l2norm_check': df_W_check['coeff_l2norm'][0],
        'df_coeff_l2norm_melt_check': df_W_check['coeff_l2norm'][1]
    }

    with open('__hypergraph__/hyperedges_weights.pickle', 'wb') as f:
        pickle.dump(dataframe_data, f)
