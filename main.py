############################################################################################
# This script performs the entire pipeline used in the paper: 
#  - Step 1: Hypergraph structure computation
#  - Step 2: Hyperedge weights computation
#  - Step 3: Post-hoc statistical analysis
#  - Step 4: Post-hoc mediation analysis
#  - Step 5: Post-hoc classification
# where the groups considered are Healthy Controls (HC), divided into HC-hypergraph and 
# HC-analysis for the hypergraph construction and for the analysis, respectively, Mild 
# Cognitive Impairment (MCI), and Alzheimer's Disease (AD) patients.
#
# The script takes as input a pickle file containing three elements:
#  - 'labels': a list/array of labels for each subject (HC, MCI, Dementia)
#  - 'fmri': a 3D numpy array of shape [num_subjs x P x N]
#  - 'demos': a pandas DataFrame containing all the information for each subject
#
# The order of the subjects in the 'labels', 'fmri', and 'demos' must be the same.
#
############################################################################################

import os
import time
import pickle
import numpy as np
import pandas as pd
from Utils.Normalization import normalize_matrix
from hypergraph_structure import hypergraph_structure
from hyperedge_weights import hyperedge_weights
from posthoc_statistical_analysis import statistical_analysis
from posthoc_mediation_analysis import mediation_analysis
from posthoc_classification import classification


##### Global Variables
PERC_HC_HYPERGRAPH = 0.7                                            # percentage of HC subjects to use for hypergraph construction
KEEP_NEGATIVE_VALUES = False                                        # if False, forse the LASSO coefficients to be positive
REGRESSION = 'lasso'                                                # regression model to use (only 'lasso' implemented)
K_CV_INNER = 5                                                      # k value for the inner k-fold cross validation    
LIST_LAMBDAS = np.round(np.arange(0.01,0.31, 0.01), 2)              # list of lambda values to test
POSITIVITY_CONSTRAINT = True                                        # if True, forse the LASSO coefficients to be positive
OPTIMAL_LAMBDA = 0.05                                               # optimal lambda value previously computed
THRESHOLD = 5.0                                                     # threshold to remove irrelevant LASSO coefficients
SCORES = ['MEM','EXF','LAN','VSP','ADAS13']                         # list of cognitive scores to consider
LIST_COVARIATES = ['Age_Visit','PTGENDER','PTEDUCAT','APOE4']       # list of covariates to include in the mediation analysis
FLAG_COVARIATES = True                                              # whether to include covariates in the mediation analysis
SCORING_METRIC = 'accuracy'                                         # scoring metric for model evaluation
TEST_SIZE = 0.2                                                     # test size for hold-out test set
K = 5                                                               # number of folds for cross-validation
RANDOM_STATE = 10                                                   # random state for reproducibility


##### Running Time
start_time = time.time()


##### Create folders for results
for folder in ['__analysis__', '__figures__', '__hypergraph__', '__res_lasso__']:
    os.makedirs(folder, exist_ok=True)


##### Load data
with open('dataset.pickle', 'rb') as f:
    data = pickle.load(f)


##### Extract data
labels = data['labels']
fmri = data['fmri']         # [fmri shape: num_subjs x P x N]
demos = data['demos']


##### Label mapping (only if labels are strings)
label_mapping = {'CN': 0, 'MCI': 1, 'Dementia': 2}
labels = np.array([label_mapping[label] for label in labels])


##### Standardize fMRI time series
fmri_norm = np.zeros_like(fmri)
for subj in range(fmri.shape[0]):
    fmri_norm[subj,:,:] = np.transpose(normalize_matrix(fmri[subj,:,:].T, by_row=True, type='zscore'))  # [P x N]


##### Select HC, MCI, and AD data, and create datasets, one for each class
idx_HC = np.where(labels == 0)[0]
idx_HC_hypergraph = idx_HC[0:int(len(idx_HC) * PERC_HC_HYPERGRAPH)]
idx_HC_analysis = idx_HC[int(len(idx_HC) * PERC_HC_HYPERGRAPH):]
idx_MCI = np.where(labels == 1)[0]
idx_AD = np.where(labels == 2)[0]

# HC-hypergraph
fmri_hc_hypergraph = fmri_norm[idx_HC_hypergraph,:]
labels_hc_hypergraph = labels[idx_HC_hypergraph]
demos_hc_hypergraph = demos.iloc[idx_HC_hypergraph,:].reset_index(drop=True)

# HC-analysis
fmri_hc_analysis = fmri_norm[idx_HC_analysis,:]
labels_hc_analysis = labels[idx_HC_analysis]
demos_hc_analysis = demos.iloc[idx_HC_analysis,:].reset_index(drop=True)

# MCI
fmri_mci = fmri_norm[idx_MCI,:]
labels_mci = labels[idx_MCI]
demos_mci = demos.iloc[idx_MCI,:].reset_index(drop=True)

# AD
fmri_ad = fmri_norm[idx_AD,:]
labels_ad = labels[idx_AD]
demos_ad = demos.iloc[idx_AD,:].reset_index(drop=True)

print('-'*140)
print(f'Info dataset                    -> # HC: {np.sum(labels == 0)} * # MCI: {np.sum(labels == 1)} * # AD: {np.sum(labels == 2)} * shape fMRI: {fmri_norm.shape}')
print(f'Info dataset HC for hypergraph  -> # HC: {np.sum(labels_hc_hypergraph == 0)} * shape fMRI: {fmri_hc_hypergraph.shape}')
print(f'Info dataset HC for analysis    -> # HC: {np.sum(labels_hc_analysis == 0)} * shape fMRI: {fmri_hc_analysis.shape}')
print(f'Info dataset MCI                -> # MCI: {np.sum(labels_mci == 1)} * shape fMRI: {fmri_mci.shape}')
print(f'Info dataset AD                 -> # AD: {np.sum(labels_ad == 2)} * shape fMRI: {fmri_ad.shape}')
print('-'*140)


##### STEP 1: HYPERGRAPH STRUCTURE COMPUTATION -------
# Compute the hypergraph structure
hypergraph_structure(fmri=fmri_hc_hypergraph, lambda_=OPTIMAL_LAMBDA, thr_perc=THRESHOLD, type_regression=REGRESSION, keep_negative=KEEP_NEGATIVE_VALUES)

# Upload hypergraph structure
with open('__hypergraph__/hypergraph_structure.pickle', 'rb') as fh:
    hypergraph = pickle.load(fh)

# Extract hypergraph structure
H = hypergraph['H']
idx_hyperedges_to_consider = hypergraph['idx_hyperedges_considered']


##### STEP 2: HYPEREDGE WEIGHTS COMPUTATION -------
# Extract the hyperedge weights
hyperedge_weights(fmri_by_group=[fmri_hc_hypergraph, fmri_hc_analysis, fmri_mci, fmri_ad], H=H, idx_hyperedges_to_consider=idx_hyperedges_to_consider, lambda_=OPTIMAL_LAMBDA, fmri_all=fmri_norm, type_regression=REGRESSION, keep_negative=KEEP_NEGATIVE_VALUES)

# Upload hyperedge weights
with open('__hypergraph__/hyperedges_weights.pickle', 'rb') as fhw:
    hyperedges_weights = pickle.load(fhw)


##### STEP 3: POST-HOC STATISTICAL ANALYSIS -------
# Compute post-hoc statistical analysis
statistical_analysis(hyperedge_weights_dict=hyperedges_weights, number_subjects_per_group=[fmri_hc_hypergraph.shape[0], fmri_hc_analysis.shape[0], fmri_mci.shape[0], fmri_ad.shape[0]])

# Upload statistical analysis results
with open('__analysis__/hyperedge_weights_statistical_analysis.pickle', 'rb') as fhsa:
    hyperedge_weights_statistical_analysis = pickle.load(fhsa)


##### STEP 4: POST-HOC MEDIATION ANALYSIS -------
# Compute post-hoc mediation analysis
demos_hc_mci_ad = pd.concat([demos_hc_analysis, demos_mci, demos_ad]).reset_index(drop=True)
# demos_hc_mci_ad['PTGENDER'] = demos_hc_mci_ad['PTGENDER'].map({'Male' : 0, 'Female' : 1})         # uncomment if sex variable contains strings

mediation_analysis(hyperedges_weights_dict=hyperedges_weights, statistical_analysis_dict=hyperedge_weights_statistical_analysis, demos=demos_hc_mci_ad, list_name_covariates=LIST_COVARIATES, list_name_scores=SCORES, covariates_analysis=FLAG_COVARIATES, random_state=RANDOM_STATE)


##### STEP 5: POST-HOC CLASSIFICATION -------
# Define labels for classification
labels_hc_ad = np.repeat([0, 1], [fmri_hc_analysis.shape[0], fmri_ad.shape[0]])
labels_mci_ad = np.repeat([0, 1], [fmri_mci.shape[0], fmri_ad.shape[0]])
labels_hc_mci = np.repeat([0, 1], [fmri_hc_analysis.shape[0], fmri_mci.shape[0]])

# Compute classification
classification(hyperedges_weights_dict=hyperedges_weights, statistical_analysis_dict=hyperedge_weights_statistical_analysis, list_labels=[labels_hc_ad, labels_mci_ad, labels_hc_mci], test_size=TEST_SIZE, k=K, scoring_metric=SCORING_METRIC, random_state=RANDOM_STATE)


##### Running Time
print('\n')
print('-'*80)
print(f'Running time: {time.time() - start_time} seconds')