import numpy as np


# Dummy solver
#===================================================================

def find_solution(A):
    '''
    Solves PageRank problem.
    Calculates lambdas ans eigenvectors,
    Finds eigenvector where lambda == 1
    Warns in case of 0 or more than 1 solutions.
    Returns -1 in case of no solution
    '''
    
    eig_vals, eig_vecs = np.linalg.eig(A)
    eig_vecs = eig_vecs.T
    ind = np.where(np.isclose(eig_vals,1))
    solutions = len(ind)  

    if solutions == 1:
        print('One solution found')
        return eig_vecs[ind[0][0]]
    
    elif solutions > 1:
        print(len(ind), 'solutions found')
        answer = []
        for i in range(solutions):
            answer.append(eig_vecs[ind[i][0]])
        
        return answer
    
    else:
        print('Solution does not exist')
        return -1

# Power method solver
#===================================================================

def power_method(A, num_simulations: int, create_history = False, autostop_threshold = 1e7):
    # Ideally choose a random vector
    # To decrease the chance that our vector
    # Is orthogonal to the eigenvector
    if create_history: 
        history_power = []
    
    b_k = np.random.rand(A.shape[1])

    for i in range(num_simulations):
        
        if create_history:
            history_power.append(b_k)

        # calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)

        # calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)

        # re normalize the vector
        new_bk = b_k1 / b_k1_norm 
        
        if autostop_threshold != 0 :
            if np.linalg.norm(b_k - new_bk) < autostop_threshold:
                return new_bk

        b_k = new_bk 
    
    return b_k



# Markov chain Monte Carlo solver
#===================================================================
def mcmc(n_trials, A):
    page = np.random.randint(A.shape[0])
    n = A.shape[0]

    freqs = np.zeros(n)

    for i in range(n_trials):  
        page = np.random.choice(np.arange(A.shape[0]).tolist() , 1, p = A[:,page].tolist())[0]
        freqs[page] += 1
    
    ans = freqs/n_trials
    ans = ans/np.linalg.norm(ans)

    return ans