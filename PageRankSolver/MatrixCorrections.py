import numpy as np

def add_weak_links(A, dampling_factor = 0.15):
    R = np.ones((A.shape)) / A.shape[0]
    A_corrected = (1-dampling_factor)*A + dampling_factor*R
    return A_corrected

def connect_dangled_node(A):
    zero_col_idxs = np.where(A.sum(0) == 0)[0]
    for _, idx in enumerate(zero_col_idxs):
        A[:,idx] = 1 / A.shape[0]
    
    return A