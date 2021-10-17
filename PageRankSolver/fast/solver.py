import numpy as np
from numba import njit

# Dummy solver
#===================================================================
@njit(fastmath = True)
def eig(A):
    
    eig_vals, eig_vecs = np.linalg.eig(A.astype(np.complex128))
    eig_vecs = eig_vecs.T
    ind = np.where(np.isclose(eig_vals,1))
    solutions = len(ind)  

    if solutions == 1:
        return eig_vecs[ind[0][0]] , 1 
    
    elif solutions > 1:
        answer = []
        for i in range(solutions):
            answer.append(eig_vecs[ind[i][0]])
        
        return answer , solutions
    
    else:
        return -1 , -1 
#===================================================================



# Power method solver
#===================================================================

@njit(fastmath=True)
def power_iteration(A, b_k):
    b_k = np.dot(A, b_k)
    b_k_norm = np.linalg.norm(b_k)
    b_k = b_k / b_k_norm
        
    return b_k
#===================================================================


# Markov chain Monte Carlo solver
#===================================================================

def choice(elements, probs_vec):
    if not np.isclose(np.sum(probs_vec), 1.0):
        raise("error")
    
    return __choice(elements, probs_vec) 


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
def mcmc(n_trials, A):
    n = A.shape[0]
    page = np.random.randint(n)

    freqs = np.zeros(n)

    for i in range(n_trials):
        page = choice(elements = np.arange(n), probs_vec= A[:,page])
        freqs[page] += 1
    ans = freqs/n_trials
    ans = ans/np.linalg.norm(ans)
    
    return ans
#===================================================================
