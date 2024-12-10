from ..base import hierarchyGraph, initialisation, training
from . import gpLVM
from gpHierarchy.base.stablePCA import StablePCA
from gpflow.utilities import set_trainable
import importlib
import tensorflow_probability, tensorflow as tf
import numpy as np

class HierarchyMIMI:
    def __init__(self, 
                 Y_0, 
                 Y_1, 
                 n_0 = 2, 
                 n_1 = 2, 
                 weightLink = 1, 
                 noise_0 = 1, 
                 noise_1 = 1, 
                 lengthscales_0 = [1,1],
                 lengthscales_1 = [1,1], 
                 fixedNoiseBetweenLevels = True, 
                 noiseVariance = 5e-2,
                 trainLengthscalesBetweenLevels = False):
        
        # Initialisation du plus bas niveau de hiérarchie :
        importlib.reload(gpLVM)
        importlib.reload(gpLVM.initialisation)
        importlib.reload(gpLVM.hierarchyGraph)
        importlib.reload(gpLVM.training)
        model_0 = gpLVM.GPLVM(Y = Y_0, 
                         n_latent_variables = n_0, 
                         omega_2_prior = noise_0,
                         gpArguments = {}, 
                         cleanNaN = True,
                         Priorlengthscale = lengthscales_0)
        self.model_0 = model_0.model
        initialisation.initialiseLatentVariablesPCA(self.model_0)
        self.modelLVM_0 = model_0
        self.Y_0 = Y_0
        self.n_0 = n_0
        self.noise_0 = noise_0
        self.noise_1 = noise_1
        self.Y_1 = Y_1
        self.n_1 = n_1
        self.ls0 = lengthscales_0
        self.ls1 = lengthscales_1
        self.noiseVariance = noiseVariance
        self.weightLink = weightLink
        self.fixedNoiseBetweenLevels = fixedNoiseBetweenLevels
        self.trainLengthscalesBetweenLevels = trainLengthscalesBetweenLevels 
        self.model_1 = hierarchyGraph.Graph()
        n_samples = len(Y_1)
        
        
    def fit(self, 
            argsOpt_0 = {'mode' : 'sequence', 'steps' : training.getTrainStrategyAlternate(2) }, 
            argsOpt_1 = {'mode' : 'sequence', 'steps' : training.getTrainStrategyAlternate(2) },):
        
        initialisation.initialiseLatentVariablesPCA(self.model_0)
        print('Training level 0...')
        self.model_0.train(**argsOpt_0)
        X_0 = self.model_0.nodes['X'].value
        n_samples = len(self.Y_1)
        with self.model_1: 
            nodeY = hierarchyGraph.ObservedVariable(self.Y_1, varianceType= 'automatic', name = 'Y', removeMean = True)
            nodeX = hierarchyGraph.LatentVariable(size = (n_samples, self.n_1), name = 'X',
                                                prior = hierarchyGraph.getGaussianPrior(self.n_1, self.noise_1))
            
            hierarchyGraph.Link(nodeX, nodeY, gp_type = 'fixedDataRegression', fixedData = X_0 / np.std(X_0, axis = 0)) 
            self.model_1.setGP(nodeX, nodeY, likelihood =  5e-2* nodeY.variance)
            gp = self.model_1.getGP(nodeY, nodeX)
            set_trainable(gp.kernel.lengthscales,self.trainLengthscalesBetweenLevels)
        initialisation.initialiseLatentVariablesPCA(self.model_1)
        self.model_1.nodes['X_1'].setValue(X_0[self.idxNotnan])
        print('Training level 1...')
        self.model_1.train(**argsOpt_1)

    def getValues(self):
        Xhierarchy = self.model_1.nodes['X'].value
        Xres = np.nan * np.ones([len(self.model_0.nodes['X'].value), Xhierarchy.shape[1]])
        Xres[self.idxNotnan] = Xhierarchy
        return self.modelMeasurements.nodes['X'].value, Xres
    
    def transform(self, Y_0_test, Y_1_test):
        pass
    
    def inverse_transform_shape(self, X_shape):
        pass