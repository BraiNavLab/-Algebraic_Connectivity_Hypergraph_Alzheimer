import itertools
import numpy as np
import pandas as pd
import networkx as nx
import pingouin as pg
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.spatial.distance import pdist
from sklearn.linear_model import Lasso, ElasticNet
from scipy.stats import shapiro, kruskal, false_discovery_control, mannwhitneyu, f_oneway, ttest_ind
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier



############################################################ General functions ############################################################
def compute_functional_connectivity(ts: np.array):
    '''
    Compute functional connectivity matrix from fMRI timeseries.\n
    Params:
        - ts (np.array): timeseries with shape [P x N], where P is timeseries length, and N is the # of ROIs [P x N]
    '''

    FC = np.corrcoef(np.transpose(ts))      # compute corrlation between all the possbile combinations of fMRI signals [N x N]

    return FC


def compute_extract_weights_values_from_datastructure(dict_hyperedges_weights: dict, idx_hyperedges: list, num_subjs: int, metric_to_extract: str):
    '''
    Function that extract the metrics computed for each hyperedge (hyperedge weight).\n
    Params:
        - dict_hyperedges_weights (dict): data structure composed of three innested dict (keys -> first level: subj_id; second level: hyperedge_id; third level: metric_name)
        - idx_hyperedges (list): list of the indices of the hyperedges to consider
        - num_subjs (int): number of subject in the datastructure
        - metric_to_extract (str): name of the metrics to extract stored in the data structure (e.g., 'algeb_conn')
    '''

    # Variable for store the weights [num_subjs x M]
    weights_hypergraph = np.zeros(shape=(num_subjs, len(idx_hyperedges)))

    # Extract values from the dict structure
    for idx_subj in range(num_subjs):
        for idx_hyperedge, hyperedge in enumerate(idx_hyperedges):
            weights_hypergraph[idx_subj,idx_hyperedge] = dict_hyperedges_weights[idx_subj][hyperedge][metric_to_extract]

    return weights_hypergraph



def compute_create_dataframe_for_analysis(data_to_merge: list, idx_hyperedges: list, group_labels: list):
    '''
    Create a pandas dataframe for the subsequent analysis and plots.\n
    Params:
        - data_to_merge (list): list of matrices for each group to merge. Each matrix has shape [num_subjs_group x len(idx_hyperedges)]
        - idx_hyperedges (list): list of the indices of the hyperedges to consider
        - group_labels (list): list of strings that represent the name of the considered groups
    '''

    data_concat = np.concatenate(data_to_merge, axis=0)

    df_data = pd.DataFrame(data=data_concat, columns=[f'e{idx}' for idx in idx_hyperedges])
    df_data_melt = pd.melt(df_data)
    num_subjs_for_groups = [data_group.shape[0] for data_group in data_to_merge]
    df_data_melt['Class'] = list(np.repeat(group_labels, num_subjs_for_groups)) * len(idx_hyperedges)

    return df_data, df_data_melt


def compute_statistical_analysis(df_data: pd.DataFrame, num_subjs_per_groups: list, metric_name: str):
    '''
    Compute the statistical analysis considering the groups provided (two or three). First check for normality and then perform Anova/Student t-test or Kruskal Wallis/Mann Whitneyu test, respectively.\n
    Params:
        - df_data (pd.DataFrame): dataframe containing the values of the metric with shape [num_subjs x M]
        - num_subjs_per_groups (list): list with the number of subjects for each group
        - metric_name (str): name of the metric in analysis for printing purpose
    '''

    # Check for normality
    print('-'*60)
    print(f'Check for: {metric_name}')
    counter_normality = 0
    for he in df_data.columns:
        test = shapiro(df_data[he].to_numpy())
        if test.pvalue > 0.05:
            counter_normality += 1
    print(f'Number of data normally distributed: {counter_normality}/{len(df_data.columns)}')
    normally_dist = counter_normality / len(df_data.columns) >= 0.5

    # Statistical analysis
    limits_groups = np.cumsum(num_subjs_per_groups)
    res_pvalues = []

    # Two groups
    if len(num_subjs_per_groups) == 2:
        for he in df_data.columns:
            # Get values for all subjects
            hyperedge_data = df_data[he].to_numpy()
            values_group_1 = hyperedge_data[0:limits_groups[0]]
            values_group_2 = hyperedge_data[limits_groups[0]:]
            
            # Compute test
            if normally_dist:
                # Two-sample t-test (parametric test)
                res_test = ttest_ind(values_group_1, values_group_2)
            else:
                # Mann Whitneyu (non-parametric test)
                res_test = mannwhitneyu(values_group_1, values_group_2)
                
            # Store results and display
            res_pvalues.append(res_test.pvalue)
            #print(f'Hyperedge {he}: {res_test}')

    # Three groups
    elif len(num_subjs_per_groups) == 3:
        for he in df_data.columns:
            # Get values for all subjects
            hyperedge_data = df_data[he].to_numpy()
            values_group_1 = hyperedge_data[0:limits_groups[0]]
            values_group_2 = hyperedge_data[limits_groups[0]:limits_groups[1]]
            values_group_3 = hyperedge_data[limits_groups[1]:]
            
            # Compute test
            if normally_dist:
                # Anova (parametric test)
                res_test = f_oneway(values_group_1, values_group_2, values_group_3)
            else:
                # Kruskal Wallis (non-parametric test)
                res_test = kruskal(values_group_1, values_group_2, values_group_3)
                
            # Store results and display
            res_pvalues.append(res_test.pvalue)
            #print(f'Hyperedge {he}: {res_test}')
    else:
        raise ValueError('The number of group for this analysis must be 2 or 3!')

    res_pvalues_np = np.array(res_pvalues)

    # Perform FDR for adjusting the p-values
    adj_pvalues = false_discovery_control(res_pvalues_np)

    print(f'\nStatistically different raw p-values: {len(res_pvalues_np[res_pvalues_np <= 0.05])}/{len(df_data.columns)}')
    print(f'Statistically different adj. p-values: {len(adj_pvalues[adj_pvalues <= 0.05])}/{len(df_data.columns)}')

    df_data_pvalues = pd.DataFrame(data=np.stack((res_pvalues_np, adj_pvalues), axis=1), columns=['raw p-value', 'adj p-value'], index=df_data.columns)
    df_data_pvalues['top'] = ['***' if pval <= 0.05 else '-' for pval in df_data_pvalues['adj p-value']]
    print('-'*60)
    
    return df_data_pvalues



def compute_statistical_analysis_groups_pairs(df_data: pd.DataFrame, df_data_pvalues: pd.DataFrame, num_subjs_per_groups: list, groups_names: list, metric_name: str):
    '''
    Compute the statistical analysis considering the groups provided (two or three). First check for normality and then perform Anova/Student t-test or Kruskal Wallis/Mann Whitneyu test, respectively.\n
    Params:
        - df_data (pd.DataFrame): dataframe containing the values of the metric with shape [num_subjs x M]
        - df_data_pvalues (pd.DataFrame): dataframe containing the results from the analysis between the three groups
        - num_subjs_per_groups (list): list with the number of subjects for each group
        - metric_name (str): name of the metric in analysis for printing purpose
        - groups_names (list): list that contains the name of the three groups
    '''    

    if len(num_subjs_per_groups) != 3: raise ValueError('The number of group for this analysis must be 3!')

    # Check for normality
    print('-'*60)
    print(f'Check for: {metric_name}')
    counter_normality = 0
    for he in df_data.columns:
        test = shapiro(df_data[he].to_numpy())
        if test.pvalue > 0.05:
            counter_normality += 1
    print(f'Number of data normally distributed: {counter_normality}/{len(df_data.columns)}')
    normally_dist = counter_normality / len(df_data.columns) >= 0.5

    # Select only the columns (hyperedges) that are statistically significant between the three groups
    df_data_subset = df_data.loc[:,df_data_pvalues[df_data_pvalues['top'] == '***'].index]

    # Statistical analysis
    limits_groups = np.cumsum(num_subjs_per_groups)
    res_pvalues_12 = []         # results from comparing group 1 and group 2
    res_pvalues_13 = []         # results from comparing group 1 and group 3
    res_pvalues_23 = []         # results from comparing group 2 and group 3

    # Three groups - statistical test between pairs of groups
    for he in df_data_subset.columns:
        # Get values for all subjects
        hyperedge_data = df_data_subset[he].to_numpy()
        values_group_1 = hyperedge_data[0:limits_groups[0]]
        values_group_2 = hyperedge_data[limits_groups[0]:limits_groups[1]]
        values_group_3 = hyperedge_data[limits_groups[1]:]
        
        # Compute test
        if normally_dist:
            # Two sample t-test (parametric test)
            res_test_12 = ttest_ind(values_group_1, values_group_2)
            res_test_13 = ttest_ind(values_group_1, values_group_3)
            res_test_23 = ttest_ind(values_group_2, values_group_3)
        else:
            # Mann Whitneyu (non-parametric test)
            res_test_12 = mannwhitneyu(values_group_1, values_group_2)
            res_test_13 = mannwhitneyu(values_group_1, values_group_3)
            res_test_23 = mannwhitneyu(values_group_2, values_group_3)
            
        # Store results and display
        res_pvalues_12.append(res_test_12.pvalue)
        res_pvalues_13.append(res_test_13.pvalue)
        res_pvalues_23.append(res_test_23.pvalue)

    res_pvalues_12_np = np.array(res_pvalues_12)
    res_pvalues_13_np = np.array(res_pvalues_13)
    res_pvalues_23_np = np.array(res_pvalues_23)

    # Perform FDR for adjusting the p-values
    adj_pvalues_12 = false_discovery_control(res_pvalues_12_np)
    adj_pvalues_13 = false_discovery_control(res_pvalues_13_np)
    adj_pvalues_23 = false_discovery_control(res_pvalues_23_np)

    # Print results
    print(f'\nGroup 1 vs 2 -> Statistically different raw p-values: {len(res_pvalues_12_np[res_pvalues_12_np <= 0.05])}/{len(df_data_subset.columns)}')
    print(f'Group 1 vs 2 -> Statistically different adj. p-values: {len(adj_pvalues_12[adj_pvalues_12 <= 0.05])}/{len(df_data_subset.columns)}\n')

    print(f'Group 1 vs 3 -> Statistically different raw p-values: {len(res_pvalues_13_np[res_pvalues_13_np <= 0.05])}/{len(df_data_subset.columns)}')
    print(f'Group 1 vs 3 -> Statistically different adj. p-values: {len(adj_pvalues_13[adj_pvalues_13 <= 0.05])}/{len(df_data_subset.columns)}\n')

    print(f'Group 2 vs 3 -> Statistically different raw p-values: {len(res_pvalues_23_np[res_pvalues_23_np <= 0.05])}/{len(df_data_subset.columns)}')
    print(f'Group 2 vs 3 -> Statistically different adj. p-values: {len(adj_pvalues_23[adj_pvalues_23 <= 0.05])}/{len(df_data_subset.columns)}\n')

    df_data_pvalues = pd.DataFrame(data=np.stack((res_pvalues_12_np, adj_pvalues_12, np.repeat(None, df_data_subset.shape[1]), res_pvalues_13_np, adj_pvalues_13, np.repeat(None, df_data_subset.shape[1]), res_pvalues_23_np, adj_pvalues_23, np.repeat(None, df_data_subset.shape[1])), axis=1), 
                                   columns=[f'raw p-value - {groups_names[0]} vs {groups_names[1]}', f'adj p-value - {groups_names[0]} vs {groups_names[1]}', f'top - {groups_names[0]} vs {groups_names[1]}',
                                            f'raw p-value - {groups_names[0]} vs {groups_names[2]}', f'adj p-value - {groups_names[0]} vs {groups_names[2]}', f'top - {groups_names[0]} vs {groups_names[2]}',
                                            f'raw p-value - {groups_names[1]} vs {groups_names[2]}', f'adj p-value - {groups_names[1]} vs {groups_names[2]}', f'top - {groups_names[1]} vs {groups_names[2]}'], 
                                   index=df_data_subset.columns)

    df_data_pvalues[f'top - {groups_names[0]} vs {groups_names[1]}'] = ['***' if pval <= 0.05 else '-' for pval in df_data_pvalues[f'adj p-value - {groups_names[0]} vs {groups_names[1]}']]
    df_data_pvalues[f'top - {groups_names[0]} vs {groups_names[2]}'] = ['***' if pval <= 0.05 else '-' for pval in df_data_pvalues[f'adj p-value - {groups_names[0]} vs {groups_names[2]}']]
    df_data_pvalues[f'top - {groups_names[1]} vs {groups_names[2]}'] = ['***' if pval <= 0.05 else '-' for pval in df_data_pvalues[f'adj p-value - {groups_names[1]} vs {groups_names[2]}']]
    print('-'*60)
    
    return df_data_pvalues


def compute_optimal_lambda(fmri: np.array, list_lambdas: list, k: int = 5, keep_negative: bool = True):
    '''
    Compute optimal lambda by inner k-fold cross validation.\n
    Params:
        - fmri (np.array): fMRI with shape [# subjs x P x N], where P is the fMRI timeseries length, and N is the number of brain region
        - list_lambda (list): list of lambda to test
        - k (int): k value for the k-fold cross validation
        - keep_negative (bool): keep the negative coefficients from LASSO for computing the incidence matrix H (default: True)
    '''

    positivity_constraint = True
    if keep_negative:
        positivity_constraint = False

    num_subj, P, N = fmri.shape
    mse_mean_total = np.zeros(shape=(num_subj,len(list_lambdas)))
    mse_std_total = np.zeros(shape=(num_subj,len(list_lambdas)))

    for idx_subj, fmri_subj in enumerate(fmri):
        #if idx_subj % 20 == 0: print(f'Computing best lambda for subject {idx_subj}/{num_subj}...')
        print(f'Computing best lambda for subject {idx_subj}/{num_subj}...')

        for idx_lambda_, lambda_ in enumerate(list_lambdas):
            #print(f'  Computing for lambda {lambda_}...')
            mse_ith_lambda_ = []

            for roi in range(N):
                X = np.copy(fmri_subj)
                # Dependent variable: i-th timeseries
                x = np.copy(X[:,roi])
                # Independent variable: all timeseries - i-th (set to zero)
                X[:,roi] = 0
                # Model definition
                model = Lasso(alpha=lambda_, max_iter=5000, fit_intercept=False, positive=positivity_constraint)
                # Cross validation
                scores = cross_val_score(model, X, x, cv=k, scoring='neg_mean_squared_error', n_jobs=16)
                mse_ith_lambda_.append(scores.mean())

                del model
            
            # Store results for a specific subject and lambda_
            mse_mean_total[idx_subj,idx_lambda_] = np.mean(mse_ith_lambda_)
            mse_std_total[idx_subj,idx_lambda_] = np.std(mse_ith_lambda_)

    return np.abs(mse_mean_total), np.abs(mse_std_total)


def compute_timeseries_relations(ts: np.array, lambda_: float = 1.0, type: str = 'lasso', keep_negative: bool = True):
    '''
    Compute sparse representation of i-th ROI with all the others (ROI=timeseries).\n
    Params:
        - ts (np.array): timeseries with shape [P x N], where P is timeseries length, and N is the # of ROIs [P x N]
        - lambda_ (float): regularization parameter
        - type (str): type of regression analysis (e.g., lasso or elastic net)
        - keep_negative (bool): keep the negative coefficients from LASSO for computing the incidence matrix H (default: True)
    '''
    
    positivity_constraint = True
    if keep_negative:
        positivity_constraint = False

    P, N = ts.shape
    alphas = []

    for roi in range(N):
        X = np.copy(ts)
        # Dependent variable: i-th timeseries
        x = np.copy(X[:,roi])
        # Independent variable: all timeseries - i-th (set to zero)
        X[:,roi] = 0
        # Define regression model
        if type == 'lasso':
            model = Lasso(alpha=lambda_, max_iter=10000, fit_intercept=False, positive=positivity_constraint)
        elif type == 'elastic-net':
            model = ElasticNet()
        else:
            raise ValueError(f'{type} regression not supported!')
        # Compute regression analysis
        model.fit(X, x)
        alphas.append(model.coef_)

    alphas = np.stack(alphas, axis=1)       # [N x N], alphas inserted by columns
    return alphas


def compute_hypergraph_structure(ts: np.array, type_regression: str = 'lasso', lambda_: float = 0.0, thr_perc: float = 25.0, keep_negative: bool = True):
    '''
    Compute the hypergraph structure for each subject.\n
    Params:
        - ts (np.array): timeseries with shape [P x N], where P is timeseries length, and N is the # of ROIs [P x N]
        - type_regression (str): type of regression analysis (e.g., lasso or elastic net) (default: 'lasso')
        - lambda_ (float): lambda value for sparsity (default: 0.0)
        - thr_perc (float): threshold in the range [0,100] to apply to the coeffiecients matrix computed (default: 25.0)
        - keep_negative (bool): keep the negative coefficients from LASSO for computing the incidence matrix H (default: True)
    '''

    # Compute relationships between timeseries
    coeff = compute_timeseries_relations(ts=ts, lambda_=lambda_, type=type_regression, keep_negative=keep_negative)
    # Remove negative, zero, and irrelevant associations
    if not keep_negative:
        coeff[coeff <= 0] = 0
        thr = np.percentile(coeff[coeff > 0], thr_perc)
        if thr < 0: thr = 0
        coeff[coeff <= thr] = 0
    else:
        thr = np.percentile(np.abs(coeff[coeff != 0]), thr_perc)
        coeff[(coeff >= -thr) & (coeff <= thr)] = 0
    # Select hyperedges that exists and with order than two (at least three nodes)
    coeff_binary = np.copy(coeff)
    coeff_binary[coeff_binary != 0] = 1
    coeff_binary[np.diag_indices_from(coeff_binary)] = 1                # insert the centroid roi (m-th) in the matrix (diagonal)
    # Define incidence matrix
    H = coeff_binary
    idx_hyperedges = np.where(np.sum(H, 0) < 3)[0]
    # Incidence matrix
    H[:,idx_hyperedges] = 0
    
    return H, coeff


def compute_sigma_Gaussian_similarity_kernel(fmri: np.array):
    '''
    Compute the optimal sigma value for the Gaussian similarity kernel.\n
    Params:
        - fmri (np.array): fMRI with shape [# subjs x P x N], where P is the fMRI timeseries length, and N is the number of brain region
    '''

    optimal_sigma_all = []

    for fmri_subj in fmri:
        pairwise_dists = pdist(np.transpose(fmri_subj), 'euclidean')
        median_dist = np.median(pairwise_dists)
        sigma = median_dist / np.sqrt(2)
        optimal_sigma_all.append(sigma)

    return np.median(optimal_sigma_all)


def compute_hyperedges_weights(fmri: np.array, H: np.array, idx_hyperedges_to_consider: np.array, optimal_lambda_: float, fmri_all: np.array, type_regression: str = 'lasso', keep_negative: bool = True):
    '''
    Compute the hyperedge weights based on a defined indicence matrix structure for each subject.\n
    Params:
        - fmri (np.array): fMRI with shape [# subjs x P x N], where P is the fMRI timeseries length, and N is the number of brain region
        - H (np.array): incidence matrix [N x M]
        - idx_hyperedges_to_consider (np.array): array that contains the indices of the hyperedges
        - optimal_lambda_ (float): optimal lambda value for LASSO regresison
        - fmri_all (np.array): fMRI from all subjects with shape [# subjs x P x N], where P is the fMRI timeseries length, and N is the number of brain region
        - type_regression (str): type of regression analysis (e.g., lasso or elastic net) (default: 'lasso')
        - keep_negative (bool): keep the negative coefficients from LASSO for computing the incidence matrix H (default: True)
    '''

    # Compute optimal sigma for Gaussian similarity kernel
    sigma = compute_sigma_Gaussian_similarity_kernel(fmri=fmri_all)

    # Create a dict to store the timeseries and related info of each hyperedge for each subject
    hyperedge_timeseries = {}
    for idx_subj in range(fmri.shape[0]):
        hyperedge_timeseries[idx_subj] = {}
        for idx_hyperedge in idx_hyperedges_to_consider:
            hyperedge_timeseries[idx_subj][idx_hyperedge] = {}

    for idx_subj, fmri_subj in enumerate(fmri):
        if idx_subj % 30 == 0: print(f'  Processing subject {idx_subj}/{fmri.shape[0]}...')
        coeff = compute_timeseries_relations(ts=fmri_subj, lambda_=optimal_lambda_, type=type_regression, keep_negative=keep_negative)
        H_coeff = coeff * H
        # Extract the timeseries for each region in all the hyperedges, for each subject
        for idx_hyperedge in idx_hyperedges_to_consider:
            hyperedge = H[:,idx_hyperedge]                                                                                              # extract the hyperedge from the incidence matrix
            coeff_hyperedge = H_coeff[:,idx_hyperedge]                                                                                  # extract the coefficients computed from LASSO for the hyperedge under analysis
            idx_brain_regions = np.where(hyperedge == 1)[0]                                                                             # isolate the brain regions that belong to that hyperedge
            ts = fmri_subj[:,idx_brain_regions]                                                                                         # extract the corresponding timeseries
            ts = np.transpose(ts)
            corr_matrix_ts = np.corrcoef(ts)                                                                                            # compute correlation matrix between the timeseries extracted
            corr_matrix_ts_abs = np.abs(corr_matrix_ts)
            G_nodes_hyperedge = nx.from_numpy_array(corr_matrix_ts_abs)
            coeff_l2norm = np.linalg.norm(coeff_hyperedge)                                                                              # compute the l2 norm over the coefficients of the LASSO
            algeb_conn = nx.linalg.algebraic_connectivity(G_nodes_hyperedge,                                                            # compute algebraic connectivity of the correlation matrix
                                                        weight='weight', 
                                                        tol=1e-10, 
                                                        method='tracemin_lu')
            Gaussian_kernel_similarity = []
            for i in range(len(idx_brain_regions)):
                for j in range(i + 1, len(idx_brain_regions)):
                    Gaussian_kernel_similarity.append(np.exp(-np.sum((ts[i,:] - ts[j,:])**2) / (2 * sigma**2)))                          # compute hyperedge weight using Gaussian-kernel

            # Store results
            hyperedge_timeseries[idx_subj][idx_hyperedge]['ts'] = ts                                        # [subset N x P]
            hyperedge_timeseries[idx_subj][idx_hyperedge]['corr_matrix'] = corr_matrix_ts
            hyperedge_timeseries[idx_subj][idx_hyperedge]['mean_corr'] = np.mean(corr_matrix_ts_abs)
            hyperedge_timeseries[idx_subj][idx_hyperedge]['algeb_conn'] = algeb_conn
            hyperedge_timeseries[idx_subj][idx_hyperedge]['Gaussian_kernel_simm'] = np.mean(Gaussian_kernel_similarity)
            hyperedge_timeseries[idx_subj][idx_hyperedge]['coeff_l2norm'] = coeff_l2norm

    return hyperedge_timeseries



def compute_fc_measures(fc: np.array, fc_structure: np.array, functional_networks: list, num_regions_fn):
    '''
    Compute the measures on FC based on a defined indicence matrix structure for each subject.\n
    Params:
        - fc (np.array): functional connectivity of all subjects [# subjs x N x N], where N is the number of brain region
        - fc_structure (np.array): average FC matrix
        - functional_networks (list): list of functional networks for each region
    '''

    # Compute binary adjacency FC matrix
    adj_fc_binary = np.copy(fc_structure.shape)
    adj_fc_binary[adj_fc_binary != 0] = 1

    # Unique functional_networks
    idx_fn = np.unique(functional_networks, return_index=True)[1]
    fn_unique = [functional_networks[idx] for idx in sorted(idx_fn)]

    # Get the indeces to define the limits for the functional networks
    idx_regions_fn_limits = np.concatenate(([0], np.cumsum(num_regions_fn)))

    # Create a dictionary with functional network name and its corresponding indeces
    idx_regions_fn_dict = {}
    for idx_fn, ith_fn in enumerate(fn_unique):
        idx_regions_fn_dict[ith_fn] = np.arange(idx_regions_fn_limits[idx_fn], idx_regions_fn_limits[idx_fn + 1])

    # Store results for all subjects
    fn_measures = {}

    for idx_subj, fc_subj in enumerate(fc):
        print(f'Processing subject: {idx_subj}/{fc.shape[0]}...')

        # Store results for a single subject    
        fn_measures_subj = {f'{i_fn_name} - {j_fn_name}':{} for idx_i_fn, i_fn_name in enumerate(fn_unique) for idx_j_fn, j_fn_name in enumerate(fn_unique) if idx_i_fn >= idx_j_fn}

        for idx_i_fn, i_fn_name in enumerate(fn_unique):
            for idx_j_fn, j_fn_name in enumerate(fn_unique):
                # if - elif - else to compute the values for all possible combinations of functional networks (like in a matrix structure -> only on the upper diagonal part (diagonal is included))
                if idx_i_fn < idx_j_fn:
                    continue
                elif idx_i_fn == idx_j_fn:
                    idx_regions_sub = idx_regions_fn_dict[i_fn_name]
                    sub_matrix = fc_subj[np.ix_(idx_regions_sub, idx_regions_sub)]
                    sub_matrix = np.abs(sub_matrix)             ############
                    mean_corr = np.mean(sub_matrix)
                    algeb_conn = nx.linalg.algebraic_connectivity(nx.from_numpy_array(sub_matrix),  # compute algebraic connectivity of the correlation matrix
                                                                weight='weight', 
                                                                tol=1e-10, 
                                                                method='tracemin_lu')
                    fn_measures_subj[f'{i_fn_name} - {j_fn_name}']['mean_corr'] = mean_corr
                    fn_measures_subj[f'{i_fn_name} - {j_fn_name}']['algeb_conn'] = algeb_conn
                else:
                    idx_regions_sub = np.concatenate((idx_regions_fn_dict[i_fn_name],idx_regions_fn_dict[j_fn_name]))
                    sub_matrix = fc_subj[np.ix_(idx_regions_sub, idx_regions_sub)]
                    sub_matrix = np.abs(sub_matrix)             ############
                    mean_corr = np.mean(sub_matrix)
                    algeb_conn = nx.linalg.algebraic_connectivity(nx.from_numpy_array(sub_matrix),  # compute algebraic connectivity of the correlation matrix
                                                                weight='weight', 
                                                                tol=1e-10, 
                                                                method='tracemin_lu')
                    fn_measures_subj[f'{i_fn_name} - {j_fn_name}']['mean_corr'] = mean_corr
                    fn_measures_subj[f'{i_fn_name} - {j_fn_name}']['algeb_conn'] = algeb_conn

        fn_measures[idx_subj] = fn_measures_subj

    return fn_measures


def cliffs_delta(x: np.array, y: np.array):
    '''
    Compute Cliff's delta effect size between two groups.\n
    Params:
        - x (np.array): values from group 1
        - y (np.array): values from group 2
    '''

    x = np.asarray(x)
    y = np.asarray(y)
    nx, ny = len(x), len(y)
    greater = np.sum(x[:, None] > y[None, :])
    less    = np.sum(x[:, None] < y[None, :])
    return (greater - less) / (nx * ny)


def global_cliffs_delta(groups: list):
    '''
    Compute Cliff's delta effect size across groups.
    Params:
        - groups (list): list of np.arrays, each array contains the values of a group
    '''

    deltas = []
    for (i, gi), (j, gj) in itertools.combinations(enumerate(groups), 2):
        d = cliffs_delta(gi, gj)
        deltas.append(abs(d))
    return np.mean(deltas)


def compute_effect_size_cliff_delta(df: pd.DataFrame, num_subjs_for_groups: list, method_name: str):
    '''
    Compute Cliff's delta effect size across groups.\n
    Params:
        - df (pd.DataFrame): dataframe containing the values of the metric with shape [num_subjs x M]
        - num_subjs_for_groups (list): list with the number of subjects for each group
        - method_name (str): name of the method used in analysis
    '''

    results_effect_sizes = []
    print(f'Effect sizes across groups for {method_name}')
    for he in df.columns:
        #print(f'  Hyperedge: {he}')
        # Example data
        group1 = df.loc[0:num_subjs_for_groups[0]-1, he].to_numpy()
        group2 = df.loc[num_subjs_for_groups[0]:num_subjs_for_groups[0]+num_subjs_for_groups[1]-1, he].to_numpy()
        group3 = df.loc[num_subjs_for_groups[0]+num_subjs_for_groups[1]:, he].to_numpy()
        #print(group1.shape, group2.shape, group3.shape)

        # Compute pairwise Cliff's deltas
        global_d = global_cliffs_delta([group1, group2, group3])
        results_effect_sizes.append(global_d)
        #print(f'  Global Cliff s δ = {global_d:.3f}')

    return results_effect_sizes

 
def compute_mediation_analysis(df_data: pd.DataFrame, df_data_pvalues: pd.DataFrame, X_values: np.array, Y_values: np.array, covariates: pd.DataFrame, flag_covariates: bool = True, random_state: int = None):
    
    
    df_data_sign = df_data.loc[:,df_data_pvalues[df_data_pvalues['top'] == '***'].index]
    covariates_names = covariates.columns.tolist()

    for e in df_data_sign.columns:
        print(f'Hyperedge {e}')

        # Get hyperedge weights
        M_np = df_data_sign.loc[:,e].to_numpy()

        # Normalize data
        X_norm = (X_values - np.mean(X_values)) / np.std(X_values)
        M_norm = (M_np - np.mean(M_np)) / np.std(M_np)
        if flag_covariates:
            cov_norm = (covariates - covariates.mean()) / covariates.std()
            cov_norm_np = cov_norm.to_numpy()
            df_data = pd.DataFrame(data=np.concatenate([X_norm.reshape(-1,1), M_norm.reshape(-1,1), Y_values.reshape(-1,1), cov_norm_np], axis=1), columns=['X', 'M', 'Y']+covariates_names)
            mediation = pg.mediation_analysis(data=df_data, x='X', m='M', y='Y', covar=covariates_names, n_boot=1000, seed=random_state)
        else:
            df_data = pd.DataFrame(data=np.concatenate([X_norm.reshape(-1,1), M_norm.reshape(-1,1), Y_values.reshape(-1,1)], axis=1), columns=['X', 'M', 'Y'])
            mediation = pg.mediation_analysis(data=df_data, x='X', m='M', y='Y', n_boot=1000, seed=random_state)
        
        display(mediation)


def compute_classification(X: np.array, y: np.array, hyperedge_names: list, test_size: float = 0.2, k: int = 5, scoring_metric: str = 'accuracy', experiment_name: str = 'Experiment', random_state: int = None):
    '''
    Perform classification using Random Forest with hyperparameter tuning via Grid Search and k-fold cross validation.\n
    Params:
        - X (np.array): Feature matrix
        - y (np.array): Labels vector
        - test_size (float): Percentage of the data to use for the test (default: 0.2)
        - k (int): Number of folds for cross validation (default: 5)
        - scoring_metric (str): Metric to use for scoring during cross validation (default: 'accuracy')
        - random_state (int): Random seed for reproducibility (default: None)
    '''
    
    # Outer K-fold cross validation for train test split
    acc = []; pre0 = []; pre1 = []; rec0 = []; rec1 = []; f10 = []; f11 = []
    pre_w = []; rec_w = []; f1_w = []

    outer_cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    for train_index, test_index in outer_cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        #print(f'Training set size: {X_train.shape[0]} subjects')
        #print(f'Test set size:     {X_test.shape[0]} subjects\n')

        # Random Forest model and hyperparameters
        model = RandomForestClassifier(random_state=random_state)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 5, 7, 10],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'bootstrap': [True, False]
        }

        # Grid search
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, return_train_score=True, cv=k, scoring=scoring_metric, n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)

        # Best parameters and best score
        #print(f'Best Parameters: {grid_search.best_params_}')
        df_res = pd.DataFrame(grid_search.cv_results_)
        train_val_res = df_res[(df_res['param_bootstrap'] == grid_search.best_params_['bootstrap']) & 
            (df_res['param_n_estimators'] == grid_search.best_params_['n_estimators']) & 
            (df_res['param_max_depth'] == grid_search.best_params_['max_depth']) & 
            (df_res['param_min_samples_split'] == grid_search.best_params_['min_samples_split']) & 
            (df_res['param_min_samples_leaf'] == grid_search.best_params_['min_samples_leaf'])][['mean_train_score','std_train_score','mean_test_score','std_test_score']]
        #print(f'\nTraining Score:   {np.round(train_val_res["mean_train_score"].values[0],3)} +/- {np.round(train_val_res["std_train_score"].values[0],3)}')
        #print(f'Validation Score: {np.round(train_val_res["mean_test_score"].values[0],3)} +/- {np.round(train_val_res["std_test_score"].values[0],3)}')

        # Evaluate on test set and Compute performance
        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        acc.append(np.round(accuracy_score(y_test, y_pred), 3))
        pre0.append(np.round(precision_score(y_test, y_pred, pos_label=0), 3))
        pre1.append(np.round(precision_score(y_test, y_pred, pos_label=1), 3))
        rec0.append(np.round(recall_score(y_test, y_pred, pos_label=0), 3))
        rec1.append(np.round(recall_score(y_test, y_pred, pos_label=1), 3))
        f10.append(np.round(f1_score(y_test, y_pred, pos_label=0), 3))
        f11.append(np.round(f1_score(y_test, y_pred, pos_label=1), 3))
        pre_w.append(np.round(precision_score(y_test, y_pred, average='weighted'), 3))
        rec_w.append(np.round(recall_score(y_test, y_pred, average='weighted'), 3))
        f1_w.append(np.round(f1_score(y_test, y_pred, average='weighted'), 3))

        print(f'Feature Importances: {best_rf.feature_importances_}')
        plt.figure(figsize=(8,3))
        sns.barplot(x=np.arange(len(hyperedge_names)), y=best_rf.feature_importances_, color='lightblue')
        plt.xticks(ticks=np.arange(len(hyperedge_names)), labels=np.array(hyperedge_names))
        plt.grid(False)
        sns.despine(top=True, right=True)
        plt.tight_layout()
        plt.savefig(f'__analysis__/{experiment_name}_feature_importances.pdf')
        plt.show()


    # Print results on test set
    print("\nClassification Report Test:")
    print(f'Accuracy Score:  {np.round(np.mean(acc), 3)} +/- {np.round(np.std(acc), 3)}')
    print(f'Precision Score: {np.round(np.mean(pre0), 3)} +/- {np.round(np.std(pre0), 3)}, {np.round(np.mean(pre1), 3)} +/- {np.round(np.std(pre1), 3)}')
    print(f'Recall Score:    {np.round(np.mean(rec0), 3)} +/- {np.round(np.std(rec0), 3)}, {np.round(np.mean(rec1), 3)} +/- {np.round(np.std(rec1), 3)}')
    print(f'F1 Score:        {np.round(np.mean(f10), 3)} +/- {np.round(np.std(f10), 3)}, {np.round(np.mean(f11), 3)} +/- {np.round(np.std(f11), 3)}\n')

    print(f'Weighted Precision Score: {np.round(np.mean(pre_w), 3)} +/- {np.round(np.std(pre_w), 3)}')
    print(f'Weighted Recall Score:    {np.round(np.mean(rec_w), 3)} +/- {np.round(np.std(rec_w), 3)}')
    print(f'Weighted F1 Score:        {np.round(np.mean(f1_w), 3)} +/- {np.round(np.std(f1_w), 3)}\n')
