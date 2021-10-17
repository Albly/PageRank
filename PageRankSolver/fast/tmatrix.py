import numpy as np
from numba import njit

# Transition matrix generators
#===================================================================

@njit(fastmath=True)
def generate(size: int, Pr = 0.5):
    mask = np.zeros((size,size))
    
    for i in range(size):
        for j in range(size):
            if i!=j:
                if np.random.binomial(1,Pr,1) == 1:
                    mask[i][j] = 1
    
    norms = np.sum(mask, axis = 0)
    norms[norms == 0] = 1
    mask = mask / norms

    return mask 


@njit(fastmath=True)
def generate_dangled(size, Pr = 0.75):
    A = generate(size, Pr)
    column = np.random.randint(0, A.shape[0])
    A[:,column] = 0
    return A


@njit(fastmath=True)
def ___generate_segmented(size, Pr):
    size_1 = np.random.randint(low = 4, high = size-4)
    size_2 = size - size_1
    
    A = np.zeros((size,size))
    
    A_part_1 = generate(size_1, Pr)
    A_part_2 = generate(size_2, Pr)


    A[0:size_1, 0:size_1] = A_part_1
    A[size_1:size_1 + size_2, size_1:size_1+size_2] = A_part_2

    return A


def generate_segmented(size, Pr = 0.75):
    if size < 10:
        raise Exception('Size {} is too small, use size more than 10'.format(size))
    
    return ___generate_segmented(size, Pr)



# Correction methods for Transition matrix
#===================================================================

@njit(fastmath = True)
def add_weak_links(A, dampling_factor = 0.15):
    R = np.ones((A.shape)) / A.shape[0]
    A_corrected = (1-dampling_factor)*A + dampling_factor*R
    return A_corrected


@njit(fastmath = True)
def connect_dangled_node(A):
    zero_col_idxs = np.where(A.sum(0) == 0)[0]
    for idx in zero_col_idxs:
        A[:,idx] = 1 / A.shape[0]
    
    return A