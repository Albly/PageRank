import plotly.figure_factory as ff
import numpy as np


def matprint(mat, fmt=".2f"):
    ''' 
    For good looking matrix printing 
    @mat - matrix for printing
    @fmt - str with format type. (.2f or .3f - best options)
    '''

    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


def plot_matrix(A):
    '''
    Visualize matrix A. 

    Doesnt work with complex values.
    Or find abs(A) for them, or
    plot separately A.real , A.imag 
    '''
    fig = ff.create_annotated_heatmap(np.around(A, decimals=2), colorscale='YlGnBu')
    fig.update_layout(
        autosize=False,
        width=500,
        height=500,
        margin=dict(
            l=5,
            r=5,
            b=5,
            t=5,
            pad=0
        ),
    )
    fig.show()