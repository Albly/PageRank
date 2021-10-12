import numpy as np

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