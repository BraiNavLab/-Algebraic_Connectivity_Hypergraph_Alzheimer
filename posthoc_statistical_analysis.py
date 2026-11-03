############################################################################################
# This script computes the statistical analysis of hyperedge weights across the three 
# groups for the four methods:
#  - Algebraic connectivity
#  - Gaussian similarity kernel
#  - Mean Pearson correlation
#  - L2-norm of the regression coefficients
#
# Then, the pairwise statistical analysis between groups is computed on the statistically
# different hyperedges FDR-corrected.
#
# The statistical analysis on hyperedge weights will be saved in the folder __analysis__
############################################################################################

import pickle
from Utils.Computation import compute_statistical_analysis, compute_statistical_analysis_groups_pairs



##### Compute statistical analysis
def statistical_analysis(hyperedge_weights_dict: dict, number_subjects_per_group: list):
    '''
    Compute post-hoc statistical analysis on hyperedge weights across groups and between control groups, and pairwise statistical analysis between group pairs.\n
    Params:
        - hyperedge_weights_dict (dict): dictionary containing the dataframes with hyperedge weights for each method
        - number_subjects_per_group (list): list with the number of subjects per group, in this order: HC-hypergraph, HC-analysis, MCI, AD
    '''
    
    list_metrics = ['algeb_conn', 'gaus_kernel', 'mean_corr', 'coeff_l2norm']
    list_full_name_metrics = ['Algebraic Connectivity', 'Gaussian Similarity Kernel', 'Mean Correlation', 'L2 norm of LASSO coefficients']

    # Computing statistical analysis across groups
    print('\nComputing statistical analysis across groups...')
    df_metric_pvalues = {metric: compute_statistical_analysis(df_data=hyperedge_weights_dict[f'df_{metric}'], num_subjs_per_groups=[number_subjects_per_group[1], number_subjects_per_group[2], number_subjects_per_group[3]], metric_name=full_name_metric) for metric, full_name_metric in zip(list_metrics, list_full_name_metrics)}

    # Computing statistical analysis between control groups (HC for hypergraph vs HC for analysis)
    print('-'*60)
    print('\nComputing statistical analysis between control groups (HC for hypergraph vs HC for analysis)...')
    df_metric_pvalues_check = {metric: compute_statistical_analysis(df_data=hyperedge_weights_dict[f'df_{metric}_check'], num_subjs_per_groups=[number_subjects_per_group[0], number_subjects_per_group[1]], metric_name=f'{full_name_metric} between control groups') for metric, full_name_metric in zip(list_metrics, list_full_name_metrics)}

    # Compute pairwise statistical analysis between group pairs
    print('-'*60)
    print('\nComputing pairwise statistical analysis between group pairs...')
    df_metric_pvalues_groups = {metric: compute_statistical_analysis_groups_pairs(df_data=hyperedge_weights_dict[f'df_{metric}'], df_data_pvalues=df_metric_pvalues[metric], num_subjs_per_groups=[number_subjects_per_group[1], number_subjects_per_group[2], number_subjects_per_group[3]], groups_names=['HC', 'MCI', 'AD'], metric_name=full_name_metric) for metric, full_name_metric in zip(list_metrics, list_full_name_metrics)}

    # Save hyperedges weights dataframes and statistical analysis results
    hyperedge_weights_statistical_analysis = {
        'df_algeb_conn_pvalues': df_metric_pvalues['algeb_conn'],
        'df_gaus_kernel_pvalues': df_metric_pvalues['gaus_kernel'],
        'df_mean_corr_pvalues': df_metric_pvalues['mean_corr'],
        'df_coeff_l2norm_pvalues': df_metric_pvalues['coeff_l2norm'],
        'df_algeb_conn_pvalues_check': df_metric_pvalues_check['algeb_conn'],
        'df_gaus_kernel_pvalues_check': df_metric_pvalues_check['gaus_kernel'],
        'df_mean_corr_pvalues_check': df_metric_pvalues_check['mean_corr'],
        'df_coeff_l2norm_pvalues_check': df_metric_pvalues_check['coeff_l2norm'],
        'df_algeb_conn_groups_pvalues': df_metric_pvalues_groups['algeb_conn'],
        'df_gaus_kernel_groups_pvalues': df_metric_pvalues_groups['gaus_kernel'],
        'df_mean_corr_groups_pvalues': df_metric_pvalues_groups['mean_corr'],
        'df_coeff_l2norm_groups_pvalues': df_metric_pvalues_groups['coeff_l2norm']
    }

    with open('__analysis__/hyperedge_weights_statistical_analysis.pickle', 'wb') as fhsa:
        pickle.dump(hyperedge_weights_statistical_analysis, fhsa)
