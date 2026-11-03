import numpy as np



def apply_normalization(matrix: np.array, type: str = 'zscore'):
    '''
    Function used to apply normalization.\n
    Params:
        - matrix (np.array): matrix to be normalized (the rows are normalized!) [N x P]
        - type (str): type of normalization. Available: zscore, min_max, max (default: 'zscore')
    '''

    matrix_norm = np.zeros_like(matrix)

    if type == 'zscore':
        matrix_norm = (matrix - np.mean(matrix)) / (np.std(matrix) + 1e-8)
    elif type == 'min_max':
        matrix_norm = (matrix - np.min(matrix)) / (np.max(matrix) - np.min(matrix) + 1e-8)
    elif type == 'max':
        matrix_norm = matrix / (np.max(matrix) + 1e-8)
    else:
        raise ValueError(f'{type} is a type of normalization not supported!')
    
    # Check for NaN -> substitute NaN with zero
    matrix_norm[np.isnan(matrix_norm)] = 0
    
    return matrix_norm



def normalize_matrix(matrix: np.array, by_row: bool = True, type: str = 'zscore'):
    '''
    Function used for normalize a matrix by row or considering the whole matrix.\n
    Params:
        - matrix (np.array): matrix to be normalized (the rows are normalized!) [N x P]
        - by_row (bool): True if normalize each row independently; False if by whole matrix (default: True)
        - type (str): type of normalization. Available: zscore, min_max, max (default: 'zscore')
    '''
    
    matrix_norm = np.zeros_like(matrix)

    if not {type}.issubset({'zscore','min_max','max'}):
        raise ValueError(f'{type} is a type of normalization not supported!')
    
    if by_row:
        for i in range(matrix.shape[0]):
            matrix_norm[i,:] = apply_normalization(matrix[i,:], type)
    else:
        matrix_norm = apply_normalization(matrix, type)

    return matrix_norm