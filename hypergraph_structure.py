############################################################################################
# This script computes the hypergraph structure for each of the HC-hypergraph subject
# using the LASSO regression with the optimal lambda. Then, the hypergraph structure and 
# informatdataion are saved. Finally, the hypergraph incidence matrix H is visualized.
#
# The hypergraph structure and data will be saved in the folder __hypergraph__
# The hypergraph image will be saved in the folder __figures__
############################################################################################

import pickle
import numpy as np
from scipy import stats
from Utils.Visualization import visualize_incidence_matrix
from Utils.Computation import compute_hypergraph_structure



##### Hypergraph construction
def hypergraph_structure(fmri: np.array, lambda_: float = 0.0, thr_perc: float = 5.0, type_regression: str = 'lasso', keep_negative: bool = False, filename_hypergraph: str = 'hypergraph'):
    '''
    Computes the hypergraph structure from fMRI time series.\n
    Params:
        - fmri (np.array): fMRI data of shape [num_subjs x P x N], where num_subjs is the number of subjects, P is the number of time points, and N is the number of brain regions
        - lambda_ (float): optimal lambda value for the regression (default: 0.0)
        - thr_perc (float): threshold percentage to apply to the coefficients (default: 5.0)
        - type_regression (str): type of regression to use ('lasso' or 'elasticnet') (default: 'lasso')
        - keep_negative (bool): whether to keep negative values in the coefficients (default: False)
        - filename_hypergraph (str): filename for saving the hypergraph structure (default: 'hypergraph')
    '''

    num_subjs_hc, P_hc, N_hc = fmri.shape
    H_hc = []
    coeff_hc = []
    print('Computing hypergraph structure for HC subjects...')
    for s in range(num_subjs_hc):
        if s % 50 == 0: print(f'  Processing subject {s}/{num_subjs_hc}...')
        # Select subject
        fmri_hc_subj = fmri[s,:]
        # Compute coefficients and incidence matrices 
        H_subj, coeff_subj = compute_hypergraph_structure(ts=fmri_hc_subj, type_regression=type_regression, lambda_=lambda_, thr_perc=thr_perc, keep_negative=keep_negative)
        # Store results
        H_hc.append(H_subj)
        coeff_hc.append(coeff_subj)

    H_hc = np.stack(H_hc)
    coeff_hc = np.stack(coeff_hc)

    # Compute mode and mean for H and coeff, respectively
    coeff = np.mean(coeff_hc, axis=0)
    H = stats.mode(H_hc, axis=0, keepdims=False).mode
    idx_hyperedges_to_remove = np.where(np.sum(H, 0) < 3)[0]    # set to zero the hyperedges that connect less than 3 regions

    # Updated mean incidence matrix
    H[:,idx_hyperedges_to_remove] = 0

    # Extract the indices of the hyperedges
    idx_hyperedges_to_consider = np.where(np.sum(H, 0) != 0)[0]
    H_final = H[:,idx_hyperedges_to_consider]
    vec_H = np.sum(H_final, axis=0)

    # Save hypergraph structure and other information
    hypergraph_data = {
        'H': H,
        'coefficients': coeff,
        'idx_hyperedges_removed': idx_hyperedges_to_remove,
        'idx_hyperedges_considered': idx_hyperedges_to_consider,
        'vec_H': vec_H,
        'parameters': {
            'REGRESSION': type_regression,
            'OPTIMAL_LAMBDA': lambda_,
            'THRESHOLD': thr_perc,
            'KEEP_NEGATIVE_VALUES': keep_negative
        }
    }
    with open('__hypergraph__/hypergraph_structure.pickle', 'wb') as f:
        pickle.dump(hypergraph_data, f)

    # Visualize incidence matrix H (hypergraph structure) and print general information
    print(f'Mean number of regions per hyperedge: {np.mean(vec_H):.2f} +/- {np.std(vec_H):.2f} [min: {np.min(vec_H)}, max: {np.max(vec_H)}]')
    visualize_incidence_matrix(H_final, filename=f'/{filename_hypergraph}_H')
