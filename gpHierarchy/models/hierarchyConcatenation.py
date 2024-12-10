from ..base.stablePCA import StablePCA
from ..base import hierarchyGraph, initialisation, training
from . import gpLVM

import numpy as np
class HierarchyConcatenation:
    def __init__(self, Y_ED, Y_meas, n_embedding_meas, n_embedding_shape, 
                 weightLink = 1, noiseShape = 1, noiseMeasurements = 1):
        
        modelLVMMeasurements = gpLVM.GPLVM(Y_meas,n_embedding_meas, noiseMeasurements)
        self.modelMeasurements = modelLVMMeasurements.model
        
        self.noiseShape = noiseShape
        self.n_embedding_shape = n_embedding_shape
        pca = StablePCA(.95)
        self.Y_ED = pca.fit_transform(Y_ED)

    def fit(self, 
            argsOptMeasurements = {'mode' : 'sequence', 'steps' : training.getTrainStrategyAlternate(2) }, 
            argsOptShape = {'mode' : 'sequence', 'steps' : training.getTrainStrategyAlternate(2) },):
        # Fit measurements
        initialisation.initialiseLatentVariablesPCA(self.modelMeasurements)
        self.modelMeasurements.train(**argsOptMeasurements)
        X_meas = self.modelMeasurements.nodes['X'].value.copy()
        
        # Fit Shape
        self.modelShape = hierarchyGraph.Graph()
        n_samples = len(self.Y_ED)
        with self.modelShape: 
            nodeY = hierarchyGraph.ObservedVariable(self.Y_ED, varianceType= 'automatic', name = 'Y', removeMean = True)

            n_samples = len(self.Y_ED)
            nodeX = hierarchyGraph.LatentVariable(size = (n_samples, self.n_embedding_shape), name = 'X',
                                                prior = hierarchyGraph.getGaussianPrior(self.n_embedding_shape, self.noiseShape))

            hierarchyGraph.Link(nodeX, nodeY, gp_type = 'fixedDataRegression', fixedData = X_meas / np.std(X_meas, axis = 0)) 
            self.modelShape.setGP(nodeX, nodeY, likelihood =  5e-2* nodeY.variance)

        initialisation.initialiseLatentVariablesPCA(self.modelShape)
        self.modelShape.train(**argsOptShape)

    def getValues(self):
        return self.modelMeasurements.nodes['X'].value, np.concatenate(
            [self.modelMeasurements.nodes['X'].value * self.modelShape.getGP('X', 'Y').scales,
             self.modelShape.nodes['X'].value], axis = 1)