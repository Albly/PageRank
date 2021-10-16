import numpy as np
from numba import njit


def generate(size, Pr = 0.5):
    '''
    Generate graph, consisted of @size nodes.
    Each node can be connected with other nodes with
    probability @Pr.
    '''
    graph = {}

    # for each node
    for i in range(size):
        connetions = []
        # look at other nodes
        for j in range(size):
            # they have to be different
            if i != j :
                # if nodes i and j are connected
                if np.random.binomial(1,Pr) == 1:
                    # add j to our list
                    connetions.append(j)
        # add list of connected nodes to dict with key @i
        graph[str(i)] = connetions

    # Calculate number of connections for each node
    g = np.zeros(size)
    for i in range(size):
        for j in range(len(graph[str(i)])):
            g[graph[str(i)][j]] +=1

    # predefine matrix 
    A = np.zeros((size,size))
    # caclculate matrix elements
    for i in range(size):
        for j in range(len(graph[str(i)])):
            A[i][graph[str(i)][j]] = 1/g[graph[str(i)][j]] 

    return A


def generate_dangled(size, Pr = 0.75):
    A = generate(size, Pr)
    column = np.random.randint(0, A.shape[0])
    A[:,column] = 0
    return A


def generate_segmented(size, Pr = 0.75):
    if size < 10:
        raise Exception('Size {} is too small, use size more than 10'.format(size))

    size_1 = np.random.randint(low = 4, high = size-4)
    size_2 = size - size_1
    
    A = np.zeros((size,size))
    
    A_part_1 = generate(size_1, Pr)
    A_part_2 = generate(size_2, Pr)


    A[0:size_1, 0:size_1] = A_part_1
    A[size_1:size_1 + size_2, size_1:size_1+size_2] = A_part_2

    return A


def check_matrix(A):
    '''
    Checks transition matrix for problems and existing solutions. 
    Returns:
     0 - if one solution is exist. 
    -1 - if generated matrix is dangled
    -2 - if it has segments
    -3 - if it has loops
    -4 - if it has no solution (Incorrecly generated matrix)
    '''

    # find eigenvalues and eigenvectors
    lambdas, vecs = np.linalg.eig(A)
    vecs = vecs.T
    lambdas = np.reshape(lambdas, -1)

    count = 0
    # go over all eigenvalues
    for _,i in enumerate(lambdas):
        # if any eigenvalue is 0, then it's dandling problem
        if i == 0:
            #print('Dandling')
            return -1
        # Calculate number of eigenvalues which are equal to 1
        elif np.isclose(i,1):
            count += 1 
    
    # If we have more then 1 eigenvalue with value 1
    # then we have unboundings
    if count > 1:
        #print('unbounding')
        return -2
    
    # If no eigenvalue with value 1 - error  
    # any column-stockastick matrix has to have such lambda
    elif count == 0:
        #print('error')
        return -4
    
    # find index where eigenvalues are equal to 1
    ind = np.where(np.isclose(lambdas,1))
    # find corresponding eigenvector
    sol = vecs[ind[0][0]]

    # if resulting eigenvector contains zeros
    idx = len(np.where(np.isclose(sol,0))[0])
    # it's loop problem
    if idx > 0 :
        #print("loop")
        return -3
    return 0


@njit(fastmath=True)
def fast_generate(size: int, Pr = 0.5):
    mask = np.zeros((size,size))
    
    for i in range(size):
        for j in range(size):
            if i!=j:
                if np.random.binomial(1,Pr,1) == 1:
                    mask[i][j] = 1
    
    norms = np.sum(mask, axis = 0)
    mask = mask / norms

    return mask 

@njit(fastmath=True)
def fast_generate_dangled(size, Pr = 0.75):
    A = fast_generate(size, Pr)
    column = np.random.randint(0, A.shape[0])
    A[:,column] = 0
    return A

@njit(fastmath=True)
def ___generate_segmented(size, Pr):
    size_1 = np.random.randint(low = 4, high = size-4)
    size_2 = size - size_1
    
    A = np.zeros((size,size))
    
    A_part_1 = fast_generate(size_1, Pr)
    A_part_2 = fast_generate(size_2, Pr)


    A[0:size_1, 0:size_1] = A_part_1
    A[size_1:size_1 + size_2, size_1:size_1+size_2] = A_part_2

    return A

def fast_generate_segmented(size, Pr = 0.75):
    if size < 10:
        raise Exception('Size {} is too small, use size more than 10'.format(size))
    
    return ___generate_segmented(size, Pr)