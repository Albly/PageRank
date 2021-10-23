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


@njit(fastmath = True)
def __choice(elements, probs_vec):
    r = np.random.rand()
    s = 0
    for i in range(len(elements)):
        s += probs_vec[i]
        if s > r:
            return elements[i]

    return elements[len(elements)]

@njit(fastmath = True)
def generate_bollobas_riordan(n,m):
     
    N = m*n                                 # Generate NxN size graph

    Trans_matrix = np.zeros((N,N))          # Non-normalized Transition matrix
    degree_vec = np.zeros((N))              # degrees of each node

    Trans_matrix[0][0] = 1                  # Add first node and link to himself 
    degree_vec[0] = 2                       # therefore its degree increase to 2

    for node in range(1,N):                 # go for each node which will be paired
        probs = np.zeros((node+1))          # vector with probabilities

        
        for pair in range(node):            # go for each existing node
            denum = 2*(node+1)-1            # denumenator of frac

            deg = degree_vec[pair]          # find degree 
            prob = deg/denum                # calculate probability

            probs[pair] = prob              # add probability to array
        
        prob = 1/denum                      # probability for node to be connected with itself
        probs[-1] = prob                    # add it to last array element 

        chosen = __choice(np.arange(node+1), probs)  # choose element randomly with given probabilities
        
        if chosen == node:                  # if the node choses itself
            degree_vec[node] +=2            # increase its degree to 2
            Trans_matrix[node][node] = 1    # add value to transition matrix
        
        else:                               # otherwise
            degree_vec[chosen] +=1          # add to chosen node 1 degree
            degree_vec[node] +=1            # and 1 degree node that chose 
            Trans_matrix[chosen][node] = 1  # add valuet to Transition matrix

    convolved_matrix = np.zeros((n,n))      # Array for convolved transition matrix

    for i in range(n):                      # for each x-cell in convolved matrix
        for j in range(n):                  # for each y-cell in convolved matrix
            x_idx, y_idx = i*m, j*m         # calculate idxs in Transition matrices
            for x_ker in range(m):          # go for each kernel x-cell
                for y_ker in range(m):      # go for each kernel y-cell 
                    convolved_matrix[i,j] += Trans_matrix[x_idx+x_ker][y_idx+y_ker] # accumulate values


    norms = np.sum(convolved_matrix, axis = 0)  # calculate 1-st norm of each column
    norms[norms == 0] = 1                       # if it is 0 then make it 1 (to avoid dividing by 0)
    
    convolved_matrix = convolved_matrix / norms # normalize convolved matrix 

    return convolved_matrix


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



@njit(fastmath = True)
def check_matrix(A, atol = 1e-12):
    '''
    Checks transition matrix for problems and existing solutions. 
    Returns:
     0 - if one solution is exist. 
     1 - if generated matrix is dangled
     2 - if it has segments
     3 - if it has loops
     4 - if it has no solution (Incorrecly generated matrix)
    '''

    problems = np.zeros(5)                                          # vector with problem idxs

    lambdas, vecs = np.linalg.eig(A.astype(np.complex128))          # find eigenvalues and eigenvectors
    vecs = vecs.T           
    lambdas = np.reshape(lambdas, -1)

    count = 0                                                       # number of probable solutions
    for i in range(len(lambdas)):                                   # go over all eigenvalues

        if np.abs(lambdas[i]) < atol:                               # check if it's equal to 0
            problems[0] = 1                                         # if any eigenvalue is 0, then it's dandling problem
        
        elif np.abs(lambdas[i] - 1) < atol:                         # if it's one
            count += 1                                              # Calculate number of eigenvalues which are equal to 1 
            
            vec = vecs[i]                                           # find it's eigenvector
            for el in vec:                                          # go for each element
                if (abs(el) < atol):                                  # if it's zero 
                    problems[2] = 1                                 # loop problem
                    break;

    if count > 1:                                                   # If we have more then 1 eigenvalue with value 1
        problems[1] = 1                                             # then we have unboundings
    

    elif count == 0:                                                # If no eigenvalue with value 1 - error  
        problems[3] = 1
    

    if np.sum(problems) == 0:
        problems[4] = 1

    return problems

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