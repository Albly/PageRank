import numpy as np
from numba import njit

# Dummy solver
#===================================================================
@njit(fastmath = True)
def eig(A):
    
    eig_vals, eig_vecs = np.linalg.eig(A.astype(np.complex128))
    eig_vecs = eig_vecs.T

    n_solutions = 0

    for lambd in eig_vals:
        if abs(lambd-1) < 1e-12:
            n_solutions +=1

    ans = np.zeros((A.shape[0], n_solutions)).astype(np.complex128)
    idx = 0

    for l_idx in range(len(eig_vals)):
        if abs(eig_vals[l_idx]-1) < 1e-12:
            ans[:,idx] = eig_vecs[l_idx,:]
    
    return ans


@njit(fastmath = True)
def max_lambdas(A, n ):
    
    eig_vals, _ = np.linalg.eig(A.astype(np.complex128))

    eig_vals = (np.abs(eig_vals).real).astype(np.float64)
    eig_vals = np.sort(eig_vals)

    ans = np.zeros((n)).astype(np.float64)

    for i in range(n):
        ans[i] = eig_vals[-i-1]
    
    return ans

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
        page = __choice(elements = np.arange(n), probs_vec= A[:,page])
        freqs[page] += 1
    ans = freqs/n_trials
    ans = ans/np.linalg.norm(ans)
    
    return ans


@njit(fastmath = True)
def mcmc_loss(n_trials, A, true_vec):
    n = A.shape[0]
    page = np.random.randint(n)
    freqs = np.zeros(n)

    loss = np.zeros(n_trials)
    ans = np.zeros(n)

    for i in range(n_trials):
        page = __choice(elements = np.arange(n), probs_vec= A[:,page])
        freqs[page] += 1
        
        ans = freqs/n_trials
        ans = ans/np.linalg.norm(ans)
        loss[i] = np.mean(((ans - true_vec)**2))

    return loss


@njit(fastmath = True)
def power(A, n_trials):
    b_k = np.random.rand(A.shape[1])

    for i in range(n_trials):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        new_bk = b_k1 / b_k1_norm 

        b_k = new_bk

    
    return b_k



@njit(fastmath = True)
def power_loss(A, n_trials, true_vec):
    b_k = np.random.rand(A.shape[1])

    loss = np.zeros(n_trials)

    for i in range(n_trials):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        new_bk = b_k1 / b_k1_norm 

        b_k = new_bk

        loss[i] = np.mean(((b_k - true_vec)**2))
    
    return loss


#===================================================================
