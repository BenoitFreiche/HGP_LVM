"""
Small dash viewer for visualizing embeddings and its reconstruction
"""

import json, os, collections, time, requests
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import numpy as np, PIL, pydicom
import plotly.graph_objs as go
import numpy as np
import logging

colors = ['red', 'blue', 'green', 'yellow', 'orange']


toDropdownOptions  = lambda v: [{'value' : vv, 'label' : vv } for vv in v]
from gpHierarchy.models import easyGP, GPC, mskr


# Add
logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.DEBUG)

import subprocess





external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets) 

# Read data.
class Data:
    def __init__(self, path, displayClosest = False, nColors = 3):
        self.i = None

        self.data = np.load(path)
        self.X = self.data['X']
        self.Y = self.data['Y']
        self.variables = {}
        for k in self.data.keys():
            if k in ['X', 'Y']:
                continue
            self.variables[k] = self.data[k]
        self.currentKey = str(list(self.variables.keys())[0])
        self.reconstructionModel = mskr.MKSRWrapper()
        self.reconstructionModel.fit(self.X, self.Y)
        
        self.syntheticPoints = {c: np.zeros(2) for c in colors[:nColors]}
        self.displayClosest = displayClosest
        
    def getRawSignal(self, i_sample):
        return self.Y[i_sample]
            
    @property
    def Y_closest(self):
        return self.Y[np.argmin(np.linalg.norm(self.X - self.syntheticPoints[colors[0]], axis = 1))]
    
    @property
    def Y_reconstructed(self):
        return self.reconstructionModel.predict([x for x in self.syntheticPoints.values()])

    @property
    def currentField(self):
        return self.variables[self.currentKey]
    
    
    def getTraces(self):
        """
        Gets the traces for the reconstruct
        """
        colors = list(self.syntheticPoints.keys())    
        Y_reconstructed = self.reconstructionModel.predict([self.syntheticPoints[c] for c in colors ])
        if self.displayClosest:
            colors += ['black']
            Y_reconstructed = np.concatenate([Y_reconstructed, self.Y_closest.reshape((1, -1))], axis = 0)
        nPointsTemporal =  self.Y.shape[1]
        t = np.arange(nPointsTemporal)
        figure = {
            'data': [
                {'x' : t, 'y' : y, 'name' : c,  'marker': dict(color= c)} for y, c in zip(Y_reconstructed, colors)
            ],
            'layout': go.Layout(
                xaxis={'title': f'Time'},
                yaxis={'title': f'Tissue speed [cm/s]'},
                hovermode='closest'
            )
        }
        return figure

    
    def getScatter(self):
        X = self.X
        z = self.currentField
        idPointSelected = None
        i,j = 0, 1
        
        m = mskr.MKSRWrapper()
        n1 = 70
        n2 = 50
        m.fit(X[z == z], z[z == z].reshape((-1, 1)))

        X_mesh = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), num = n1)
        Y_mesh = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), num = n2)
        XX , YY = np.meshgrid(X_mesh,  Y_mesh)
        XY = np.stack((XX.flatten(), YY.flatten()), axis = 1)
        Z = m.predict(XY).reshape((n2, n1 ))

        if idPointSelected is None:
            idPointSelected = -1
        figure = {
            'data': [
                go.Scatter(
                    x=X[:, i],
                    y=X[:, j],
                    marker =dict(color= ['red' if ii == idPointSelected else ' black' for ii in range(X.shape[0])]),
                    text = np.arange(len(X)),
                    mode='markers',
                    opacity=0.95,
                )] +   [go.Scatter(
                    x=[x[i]],
                    y=[x[j]],
                    marker =dict(color= c),
                    mode='markers',
                    opacity=0.95,
                ) for c,x in self.syntheticPoints.items()]  + [
                                go.Contour(
                    z= Z,
                    x= X_mesh, # horizontal axis
                    y= Y_mesh, # vertical axis
                    opacity = 0.5
                )]
,
            
            'layout': go.Layout(
                width = 600,
                height = 600,
                xaxis={'title': f'Dimension {i}', 'showgrid' : False, 'zeroline': False,},
                yaxis={'title': f'Dimension {j}', 'scaleanchor' : "x", 'scaleratio' : 1, 'showgrid' : False, 'zeroline': False},
                hovermode='closest',
                showlegend=False,
                title=self.currentKey,
            )
        }
        return figure

data = Data('../experiments/supervisedHierarchy/dataTDI_fpca.npz')

divEmbeddingSelector = html.Div(children = [
    html.Div(
        [dcc.Dropdown(id = 'dropdownFieldSelection'  ,options = [{'value' : s, 'label': s} for  s in data.variables.keys()],  value = data.currentKey,  className = "three columns"),
        dcc.Dropdown(id = 'dropdownColorSelection'  ,options = toDropdownOptions(data.syntheticPoints.keys()),  value = colors[0],  className = "three columns")],
        className = 'row'
    )
    ]
)
## On click on points, or change the rawData used or the latentVariable
#@app.callback(
#    Output('dropdownPointSelector', 'value'),
#    [Input('graphScatter', 'clickData')]
#)
#def onclick(clicks):
#    if clicks is not None:
#        return clicks['points'][0]['text'] #clicks['points'][0]['pointIndex']
#    else:
#        raise dash.exceptions.PreventUpdate

@app.callback(
    [Output('graphTrace', 'figure'),  Output('graphScatter', 'figure')],
    [Input('graphScatter', 'clickData'), Input('dropdownFieldSelection', 'value') , Input('dropdownColorSelection', 'value')]
)
def displayTrace(clicks, newField, colorSyntheticPoint):
    """
    Fucking messs.... need to make it more organized and efficient
    TODO: reconstruction check with parameters, only if some changed...
    """
    if clicks is not None:
        print(clicks)
        p = clicks['points'][0]
        data.syntheticPoints[colorSyntheticPoint] = np.array([p['x'], p['y']])
    data.currentKey = newField
    return data.getTraces(), data.getScatter()




app.layout = html.Div(children = [divEmbeddingSelector,
    html.Div(children = [
        html.Div(dcc.Graph( id = 'graphScatter',
                           figure = {}
        ), className = "six columns ", 
                 style = {    'display': 'table-cell',
                        'vertical-align': 'middle'}),
        
        html.Div(dcc.Graph( id = 'graphTrace',
                   figure = {},
        ), className = "six columns ", 
                 style = {    'display': 'table-cell',
                        'vertical-align': 'middle'}),

    ])
])


if __name__ == '__main__':
    app.run_server(debug=True)
