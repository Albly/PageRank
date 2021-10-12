import numpy as np

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
