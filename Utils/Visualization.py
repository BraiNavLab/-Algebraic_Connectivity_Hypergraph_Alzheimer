import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context='paper', style='whitegrid')



def visualize_incidence_matrix(H: np.array, colormap: str = 'Grays', filename: str = 'hypergraph'):
    '''
    Function to visualize the incidence matrix H of the hypergraph.\n
    Params:
        - H (np.array): incidence matrix [N x M]
        - colormap (str): colormap to use for the visualization (default: 'Grays')
        - filename (str): name of the file to save the figure (default: 'hypergraph')
    '''

    plt.figure(figsize=(6,6))
    sns.heatmap(H, cmap=colormap, cbar=False, square=True)
    plt.xlabel('Hyperedges', fontdict={'size':12})
    plt.ylabel('Brain regions', fontdict={'size':12})
    plt.xticks(np.arange(0,H.shape[1],8), [f'e{e}' for e in np.arange(0,H.shape[1],8)], rotation=0, fontsize=8)
    plt.yticks(np.arange(0,H.shape[0],10), np.arange(0,H.shape[0],10), fontsize=8)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f'__figures__/{filename}.pdf')
    plt.show()


def visualize_barplot_best_lambda(results_k_foldCV: np.array, list_lambdas: list, k: int, filename: str):
    '''
    Visualize the results obtained in the script for computing the optimal lambda for the LASSO in hypergraph.\n
    Params:
        - results_k_foldCV (np.array): numpy array with shape [num_subj x num_lambdas_tested], where each cell of this matrix is filled with mean MSE (from k-fold CV) value for that specific subject and lambda
        - list_lambda (list): list of lambda tested
        - k (int): k value used in k-fold cross validation
        - filename (str): path and name of the figure to save (without .format -> default in PDF)
    '''
    
    # Find the lambda with lowest MSE (Optimal lambda)
    color_palette = np.repeat('#E1E1E1', len(list_lambdas))
    color_palette[np.argmin(results_k_foldCV.mean(0))] = '#82DDF0'      # light green: #90E39A; light blue: #82DDF0; light red: #F03A47

    # Create dataframe to plot the data
    df_subj_lambda_mse = pd.DataFrame(data=results_k_foldCV,columns=list_lambdas)

    # Display results in a barplot
    fig, ax = plt.subplots(1, 1, figsize=(12,5))
    sns.barplot(data=df_subj_lambda_mse, errorbar='ci', palette=list(color_palette), alpha=0.75, ax=ax)

    ax.set_xlabel('Lambda values')
    ax.set_ylabel(f'Mean MSE on validation (inner {k}-fold CV)')

    plt.tight_layout()
    plt.savefig(f'__figures__/{filename}.pdf', dpi=600)
    plt.show()
    plt.close()
