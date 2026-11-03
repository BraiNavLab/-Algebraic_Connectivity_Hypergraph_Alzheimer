############################################################################################
# This script computes the three binary classification tasks 
#  - HC vs AD
#  - MCI vs AD
#  - HC vs MCI
# using the hyperedge weights as features for the four methods.
#
# Random Forest classifier is used with nested 5-fold cross-validation for tuning the 
# hyperparameters and then tested on a hold-out test set.
#
# Finally, the features importance is computed and saved for each classification task.
#
# The features importance figures will be saved in the folder __figures__
############################################################################################

import numpy as np
from Utils.Computation import compute_classification



##### Compute classification
def classification(hyperedges_weights_dict: dict, statistical_analysis_dict: dict, list_labels: list, test_size=0.2, k=5, scoring_metric='roc_auc', random_state=10):
    '''
    Computes the three binary classification tasks using the hyperedge weights as features.\n
    Params:
        - hyperedges_weights_dict (dict): dictionary containing the hyperedges weights for each method and each group
        - statistical_analysis_dict (dict): dictionary containing the statistical analysis results for each method
        - list_labels (list): list containing the labels arrays for the classification tasks (pos. 1: HC vs AD, pos. 2: MCI vs AD, pos. 3: HC vs MCI)
        - test_size (float): proportion of the dataset to include in the test split
        - k (int): number of folds for cross-validation
        - scoring_metric (str): metric to optimize during hyperparameter tuning
        - random_state (int): random seed for reproducibility
    '''

    print('\n\nClassification Tasks...')

    ############################################## HC vs AD ##############################################
    # Collect data for HC and AD
    algeb_conn_hc_ad = np.vstack((hyperedges_weights_dict['algeb_conn_np']['hc_analysis'], hyperedges_weights_dict['algeb_conn_np']['ad']))
    gaus_kernel_hc_ad = np.vstack((hyperedges_weights_dict['gaus_kernel_np']['hc_analysis'], hyperedges_weights_dict['gaus_kernel_np']['ad']))
    mean_corr_hc_ad = np.vstack((hyperedges_weights_dict['mean_corr_np']['hc_analysis'], hyperedges_weights_dict['mean_corr_np']['ad']))
    coeff_l2norm_hc_ad = np.vstack((hyperedges_weights_dict['coeff_l2norm_np']['hc_analysis'], hyperedges_weights_dict['coeff_l2norm_np']['ad']))

    # Algebraic Connectivity
    print('\nAlgebraic Connectivity: HC vs AD ---------------')
    algeb_conn_norm = (algeb_conn_hc_ad - algeb_conn_hc_ad.mean()) / algeb_conn_hc_ad.std()
    hyperedge_names_algeb_conn = statistical_analysis_dict['df_algeb_conn_pvalues'].index.tolist()

    compute_classification(X=algeb_conn_norm, y=list_labels[0], hyperedge_names=hyperedge_names_algeb_conn, test_size=test_size, k=k, scoring_metric=scoring_metric, random_state=random_state, experiment_name='Algebraic_Connectivity_HC_vs_AD')

    # Gaussian Similarity Kernel 
    print('\nGaussian Similarity Kernel: HC vs AD ---------------')
    gaus_kernel_norm = (gaus_kernel_hc_ad - gaus_kernel_hc_ad.mean()) / gaus_kernel_hc_ad.std()
    hyperedge_names_gaus_kernel = statistical_analysis_dict['df_gaus_kernel_pvalues'].index.tolist()

    compute_classification(X=gaus_kernel_norm, y=list_labels[0], hyperedge_names=hyperedge_names_gaus_kernel, test_size=test_size, k=k, scoring_metric=scoring_metric, random_state=random_state, experiment_name='Gaussian_Similarity_Kernel_HC_vs_AD')

    # Mean Pearson Correlation
    print('\nMean Pearson Correlation: HC vs AD ---------------')
    mean_corr_norm = (mean_corr_hc_ad - mean_corr_hc_ad.mean()) / mean_corr_hc_ad.std()
    hyperedge_names_mean_corr = statistical_analysis_dict['df_mean_corr_pvalues'].index.tolist()

    compute_classification(X=mean_corr_norm, y=list_labels[0], hyperedge_names=hyperedge_names_mean_corr, test_size=test_size, k=k, scoring_metric=scoring_metric, random_state=random_state, experiment_name='Mean_Pearson_Correlation_HC_vs_AD')

    # L2 norm of LASSO coefficients
    print('\nL2 norm of LASSO coefficients: HC vs AD ---------------')
    coeff_l2norm_norm = (coeff_l2norm_hc_ad - coeff_l2norm_hc_ad.mean()) / coeff_l2norm_hc_ad.std()
    hyperedge_names_coeff_l2norm = statistical_analysis_dict['df_coeff_l2norm_pvalues'].index.tolist()

    compute_classification(X=coeff_l2norm_norm, y=list_labels[0], hyperedge_names=hyperedge_names_coeff_l2norm, test_size=test_size, k=k, scoring_metric=scoring_metric, random_state=random_state, experiment_name='L2_Norm_LASSO_Coefficients_HC_vs_AD')


    ############################################## MCI vs AD ##############################################
    # Collect data for MCI and AD
    algeb_conn_mci_ad = np.vstack((hyperedges_weights_dict['algeb_conn_np']['mci'], hyperedges_weights_dict['algeb_conn_np']['ad']))
    gaus_kernel_mci_ad = np.vstack((hyperedges_weights_dict['gaus_kernel_np']['mci'], hyperedges_weights_dict['gaus_kernel_np']['ad']))
    mean_corr_mci_ad = np.vstack((hyperedges_weights_dict['mean_corr_np']['mci'], hyperedges_weights_dict['mean_corr_np']['ad']))
    coeff_l2norm_mci_ad = np.vstack((hyperedges_weights_dict['coeff_l2norm_np']['mci'], hyperedges_weights_dict['coeff_l2norm_np']['ad']))

    # Algebraic Connectivity
    print('\nAlgebraic Connectivity: MCI vs AD ---------------')
    algeb_conn_norm = (algeb_conn_mci_ad - algeb_conn_mci_ad.mean()) / algeb_conn_mci_ad.std()

    compute_classification(X=algeb_conn_norm, y=list_labels[1], hyperedge_names=hyperedge_names_algeb_conn, test_size=test_size, k=k, scoring_metric=scoring_metric, random_state=random_state, experiment_name='Algebraic_Connectivity_MCI_vs_AD')

    # Gaussian Similarity Kernel 
    print('\nGaussian Similarity Kernel: MCI vs AD ---------------')
    gaus_kernel_norm = (gaus_kernel_mci_ad - gaus_kernel_mci_ad.mean()) / gaus_kernel_mci_ad.std()

    compute_classification(X=gaus_kernel_norm, y=list_labels[1], hyperedge_names=hyperedge_names_gaus_kernel, test_size=test_size, k=k, scoring_metric=scoring_metric, random_state=random_state, experiment_name='Gaussian_Similarity_Kernel_MCI_vs_AD')

    # Mean Pearson Correlation
    print('\nMean Pearson Correlation: MCI vs AD ---------------')
    mean_corr_norm = (mean_corr_mci_ad - mean_corr_mci_ad.mean()) / mean_corr_mci_ad.std()

    compute_classification(X=mean_corr_norm, y=list_labels[1], hyperedge_names=hyperedge_names_mean_corr, test_size=test_size, k=k, scoring_metric=scoring_metric, random_state=random_state, experiment_name='Mean_Pearson_Correlation_MCI_vs_AD')

    # L2 norm of LASSO coefficients
    print('\nL2 norm of LASSO coefficients: MCI vs AD ---------------')
    coeff_l2norm_norm = (coeff_l2norm_mci_ad - coeff_l2norm_mci_ad.mean()) / coeff_l2norm_mci_ad.std()

    compute_classification(X=coeff_l2norm_norm, y=list_labels[1], hyperedge_names=hyperedge_names_coeff_l2norm, test_size=test_size, k=k, scoring_metric=scoring_metric, random_state=random_state, experiment_name='L2_Norm_LASSO_Coefficients_MCI_vs_AD')


    ############################################## HC vs MCI ##############################################
    # Collect data for HC and MCI
    algeb_conn_hc_mci = np.vstack((hyperedges_weights_dict['algeb_conn_np']['hc_analysis'], hyperedges_weights_dict['algeb_conn_np']['mci']))
    gaus_kernel_hc_mci = np.vstack((hyperedges_weights_dict['gaus_kernel_np']['hc_analysis'], hyperedges_weights_dict['gaus_kernel_np']['mci']))
    mean_corr_hc_mci = np.vstack((hyperedges_weights_dict['mean_corr_np']['hc_analysis'], hyperedges_weights_dict['mean_corr_np']['mci']))
    coeff_l2norm_hc_mci = np.vstack((hyperedges_weights_dict['algeb_conn_np']['hc_analysis'], hyperedges_weights_dict['algeb_conn_np']['mci']))

    # Algebraic Connectivity
    print('\nAlgebraic Connectivity: HC vs MCI ---------------')
    algeb_conn_norm = (algeb_conn_hc_mci - algeb_conn_hc_mci.mean()) / algeb_conn_hc_mci.std()

    compute_classification(X=algeb_conn_norm, y=list_labels[2], hyperedge_names=hyperedge_names_algeb_conn, test_size=test_size, k=k, scoring_metric=scoring_metric, random_state=random_state, experiment_name='Algebraic_Connectivity_HC_vs_MCI')

    # Gaussian Similarity Kernel 
    print('\nGaussian Similarity Kernel: HC vs MCI ---------------')
    gaus_kernel_norm = (gaus_kernel_hc_mci - gaus_kernel_hc_mci.mean()) / gaus_kernel_hc_mci.std()

    compute_classification(X=gaus_kernel_norm, y=list_labels[2], hyperedge_names=hyperedge_names_gaus_kernel, test_size=test_size, k=k, scoring_metric=scoring_metric, random_state=random_state, experiment_name='Gaussian_Similarity_Kernel_HC_vs_MCI')

    # Mean Pearson Correlation
    print('\nMean Pearson Correlation: HC vs MCI ---------------')
    mean_corr_norm = (mean_corr_hc_mci - mean_corr_hc_mci.mean()) / mean_corr_hc_mci.std()

    compute_classification(X=mean_corr_norm, y=list_labels[2], hyperedge_names=hyperedge_names_mean_corr, test_size=test_size, k=k, scoring_metric=scoring_metric, random_state=random_state, experiment_name='Mean_Pearson_Correlation_HC_vs_MCI')

    # L2 norm of LASSO coefficients
    print('\nL2 norm of LASSO coefficients: HC vs MCI ---------------')
    coeff_l2norm_norm = (coeff_l2norm_hc_mci - coeff_l2norm_hc_mci.mean()) / coeff_l2norm_hc_mci.std()

    compute_classification(X=coeff_l2norm_norm, y=list_labels[2], hyperedge_names=hyperedge_names_coeff_l2norm, test_size=test_size, k=k, scoring_metric=scoring_metric, random_state=random_state, experiment_name='L2_Norm_LASSO_Coefficients_HC_vs_MCI')
