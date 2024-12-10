from ..base import hierarchyGraph, initialisation, training
from . import gpLVM
from gpHierarchy.base.stablePCA import StablePCA
from gpflow.utilities import set_trainable

import tensorflow_probability, tensorflow as tf
import numpy as np

class HierarchyPrior:
    def __init__(self, Y_ED, Y_meas, n_embedding_meas, n_embedding_shape, 
                 weightLink = 1, noiseShape = 1, noiseMeasurements = 1, fixedNoiseBetweenLevels = True, noiseVariance = 5e-2,
                 trainLengthscalesBetweenLevels = False,
                 paramsShapeModel = {}):
        
        modelLVMMeasurements = gpLVM.GPLVM(Y_meas,n_embedding_meas, noiseMeasurements)
        self.modelMeasurements = modelLVMMeasurements.model
        self.modelLVMMeasurements = modelLVMMeasurements

        self.Y_ED = Y_ED
        self.n_embedding_shape = n_embedding_shape
        self.noiseShape = noiseShape
        
        self.noiseVariance = noiseVariance
        self.weightLink = weightLink
        self.fixedNoiseBetweenLevels = fixedNoiseBetweenLevels
        
        modelLVMShape = gpLVM.GPLVM(self.Y_ED,self.n_embedding_shape, self.noiseShape, dataTransformer = StablePCA(.95))
        self.modelLVMShape = modelLVMShape
        self.modelShape = modelLVMShape.model
        self.idxNotnan = modelLVMShape.idxNotnan
        
        # Add the link
        with self.modelShape: 
            X_meas_node = hierarchyGraph.ObservedVariable(self.modelMeasurements.nodes['X'].value[self.idxNotnan], name = 'X_meas')
            hierarchyGraph.Link(X_meas_node, self.modelShape.nodes['X'], weight = self.weightLink) 
            self.modelShape.setGP(X_meas_node, self.modelShape.nodes['X'], likelihood =  self.noiseVariance)
            gp = self.modelShape.getGP(X_meas_node, self.modelShape.nodes['X'])
            set_trainable(gp.kernel.lengthscales,trainLengthscalesBetweenLevels)
            if self.fixedNoiseBetweenLevels:
                set_trainable(gp.likelihood.variance, False)
            #set_trainable(gp.kernel.variance, True)
            #self.modelShape.getGP(X_meas, self.modelShape.nodes['X']).likelihood.variance.prior = tensorflow_probability.distributions.HalfNormal(
            #                tf.constant(0.25, dtype = tf.float64))
  
        
    def fit(self, 
            argsOptMeasurements = {'mode' : 'sequence', 'steps' : training.getTrainStrategyAlternate(2) }, 
            argsOptShape = {'mode' : 'sequence', 'steps' : training.getTrainStrategyAlternate(2) },):
        # Fit measurements
        initialisation.initialiseLatentVariablesPCA(self.modelMeasurements)
        
        self.modelMeasurements.train(**argsOptMeasurements)
        X_meas = self.modelMeasurements.nodes['X'].value
        
        self.modelShape.nodes['X_meas'].setValue(X_meas[self.idxNotnan])

        # Fit Shape
        initialisation.initialiseLatentVariablesPCA(self.modelShape)
        self.modelShape.train(**argsOptShape)

    def getValues(self):
        Xhierarchy = self.modelShape.nodes['X'].value
        Xres = np.nan * np.ones([len(self.modelMeasurements.nodes['X'].value), Xhierarchy.shape[1]])
        Xres[self.idxNotnan] = Xhierarchy
        return self.modelMeasurements.nodes['X'].value, Xres
    
    def transform(self, Y_ED_test, Y_meas_test):
        pass
    
    def inverse_transform_shape(self, X_shape):
        pass