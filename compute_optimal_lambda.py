############################################################################################
# This script computes the LASSO regression model for each of the HC-hypergraph subject
# in order to find the optimal lambda value via k-fold cross-validation that minimizes
# the MSE. In addition, the script will save the image representing the MSE values for 
# each lambda value tested in a barplot.
#
# The MSE values will be saved in the folder __res_lasso__
# The image will be saved in the folder __figures__
############################################################################################

import os
import pickle
import time
import numpy as np
from Utils.Visualization import visualize_barplot_best_lambda
from Utils.Normalization import normalize_matrix
from Utils.Computation import compute_optimal_lambda



##### Running Time
start_time = time.time()


##### Global Variables
KEEP_NEGATIVE_VALUES = False                                # if False, forse the LASSO coefficients to be positive
REGRESSION = 'lasso'                                        # regression model to use (only 'lasso' implemented)
K_CV_INNER = 5                                              # k value for the inner k-fold cross validation    
LIST_LAMBDAS = np.round(np.arange(0.01,0.31, 0.01), 2)      # list of lambda values to test
PERC_HC_HYPERGRAPH = 0.7                                    # percentage of HC subjects to use for hypergraph construction


##### Create folders for results
for folder in ['__analysis__', '__figures__', '__hypergraph__', '__res_lasso__']:
    os.makedirs(folder, exist_ok=True)
    

##### Load data
with open('data/dataset.pickle', 'rb') as f:    
    data = pickle.load(f)


##### Extract data
labels = data['labels']
fmri = data['fmri']         # [fmri shape: num_subjs x P x N]
demos = data['demos']


##### Label mapping (only if labels are strings)
label_mapping = {'HC': 0, 'MCI': 1, 'Dementia': 2}
labels = np.array([label_mapping[label] for label in labels])


##### Standardize fMRI time series
fmri_norm = np.zeros_like(fmri)
for subj in range(fmri.shape[0]):
    fmri_norm[subj,:,:] = np.transpose(normalize_matrix(fmri[subj,:,:].T, by_row=True, type='zscore'))  # [P x N]


##### Select 70% of the HC in order to create the HC-hypergraph group
idx_HC = np.where(labels == 0)[0]
idx_HC_hypergraph = idx_HC[0:int(len(idx_HC) * PERC_HC_HYPERGRAPH)]
idx_HC_analysis = idx_HC[int(len(idx_HC) * PERC_HC_HYPERGRAPH):]

fmri_hc_hypergraph = fmri_norm[idx_HC_hypergraph,:]
labels_hc_hypergraph = labels[idx_HC_hypergraph]
demos_hc_hypergraph = demos.iloc[idx_HC_hypergraph,:].reset_index(drop=True)

print(f'Info dataset HC for hypergraph  -> # HC: {np.sum(labels_hc_hypergraph == 0)} * shape fMRI: {fmri_hc_hypergraph.shape}\n')


##### Find optimal lambda value for hypergraph construction
mse_mean, mse_std = compute_optimal_lambda(fmri_hc_hypergraph, LIST_LAMBDAS, k=K_CV_INNER, keep_negative=KEEP_NEGATIVE_VALUES)

np.save(f'__res_lasso__/mse_mean_optimal_lambda_inner_{K_CV_INNER}-foldCV_negative-coeff-{KEEP_NEGATIVE_VALUES}.npy', mse_mean)
np.save(f'__res_lasso__/mse_std_optimal_lambda_inner_{K_CV_INNER}-foldCV_negative-coeff-{KEEP_NEGATIVE_VALUES}.npy', mse_std)


##### Visualize barplot with lambda values and highlighted the optimal lambda
visualize_barplot_best_lambda(results_k_foldCV=mse_mean, list_lambdas=LIST_LAMBDAS, k=K_CV_INNER, filename=f'Barplot_lambdas_{K_CV_INNER}-fold_CV')


##### Running Time
print(f'Running time: {time.time() - start_time} seconds')