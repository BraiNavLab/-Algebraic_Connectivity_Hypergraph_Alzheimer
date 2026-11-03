############################################################################################
# This script computes the mediation analysis for the four methods:
#  - Algebraic connectivity
#  - Gaussian similarity kernel
#  - Mean Pearson correlation
#  - L2-norm of the regression coefficients 
# 
# The mediation analysis can be performed including or not covariates.
#
# The results are displayed in the console.
############################################################################################

import pandas as pd
from Utils.Computation import compute_mediation_analysis



##### Mediation analysis
def mediation_analysis(hyperedges_weights_dict: dict, statistical_analysis_dict: dict, demos: pd.DataFrame, list_name_covariates: list, list_name_scores: list, covariates_analysis: bool = True, random_state: int = 10):
    '''
    Compute post-hoc mediation analysis on hyperedge weights for each method.\n
    Params:
        - hyperedges_weights_dict (dict): dictionary containing the dataframes with hyperedge weights for each method
        - statistical_analysis_dict (dict): dictionary containing the dataframes with the statistical analysis results for each method
        - demos (pd.DataFrame): dataframe containing the demographic, clinical, scores, and covariates data for HC-analysis, MCI and AD subjects
        - list_name_covariates (list): list with the names of the covariates to include in the analysis
        - list_name_scores (list): list with the names of the cognitive scores to include in the analysis
        - covariates_analysis (bool): whether to include covariates in the mediation analysis (default: True)
        - random_state (int): random state for reproducibility (default: 10)
    '''

    X = demos['TAU_CTX_ENTORHINAL_SUVR'].to_numpy()

    covariates = demos[list_name_covariates]

    for metric, full_metric_name in zip(['algeb_conn', 'gaus_kernel', 'mean_corr', 'coeff_l2norm'], ['Algebraic Connectivity', 'Gaussian Similarity Kernel', 'Mean Pearson Correlation', 'L2 norm of LASSO coefficients']):
        print('-'*80)
        print(f'Mediation Analysis: {full_metric_name}...')

        for cognitive_variable in list_name_scores:
            print(f'{cognitive_variable} ************************************')
            
            Y = demos[cognitive_variable].to_numpy()
            compute_mediation_analysis(df_data=hyperedges_weights_dict[f'df_{metric}'], df_data_pvalues=statistical_analysis_dict[f'df_{metric}_pvalues'], X_values=X, Y_values=Y, covariates=covariates, flag_covariates=covariates_analysis, random_state=random_state)