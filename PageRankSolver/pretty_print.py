import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go


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


def plot_matrix(A, width = 300, height = 300, cmap = 'YlGnBu'):
    '''
    Visualize matrix A. 

    Doesnt work with complex values.
    Or find abs(A) for them, or
    plot separately A.real , A.imag 
    '''

    if A.shape[0] < 20:
        fig = ff.create_annotated_heatmap(np.around(A, decimals=2), colorscale = cmap)
        fig.update_layout(
            autosize=False,
            width = width,
            height = height,
            margin=dict(
                l=5,
                r=5,
                b=5,
                t=5,
                pad=0
            ),
        )
    else: 
        data = [go.Heatmap(
             z=A, 
             colorscale = cmap)]

        layout = go.Layout(autosize = False, template='none', width=width, height=height,margin=go.layout.Margin(l=40, r=40, b=40, t=40,))     
        fig = go.Figure(data=data, layout=layout)
        
    fig['layout']['yaxis']['autorange'] = "reversed"
    fig.show()