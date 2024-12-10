from ..base import hierarchyGraph, initialisation, training
from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf
import numpy as np
from time import time

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, input_array, y=None):
        return self
    
    def transform(self, input_array, y=None):
        return input_array*1
    def inverse_transform(self, input_array, y=None):
        return input_array*1

class GPLVM:
    def __init__(self, Y, n_latent_variables, 
                 omega_2_prior = 1,
                 dataTransformer = IdentityTransformer(),
                 gpArguments = {},
                 directory = '',
                 cleanNaN = True,
                 color = None,
                 color2 = None,
                 Priorlengthscale = None,
                 title=''):
        self.idxNotnan = Y[:,0] == Y[:,0]
        self.nSamples = len(Y)
        self.cleanNaN = cleanNaN
        self.Priorlengthscale = Priorlengthscale
        
        if cleanNaN:
            Y = Y[self.idxNotnan]

        self.dataTransformer =dataTransformer
        Y_red = self.dataTransformer.fit_transform(Y)

        self.model = hierarchyGraph.Graph()

        with self.model: 

            nodeY = hierarchyGraph.ObservedVariable(Y_red[:, :], varianceType= 1, name = 'Y', removeMean = False)

            n_samples = len(Y_red)
            nodeX = hierarchyGraph.LatentVariable(size = (n_samples , n_latent_variables), name = 'X',
                                                prior = hierarchyGraph.getGaussianPrior(n_latent_variables, omega_2_prior))

            hierarchyGraph.Link(nodeX, nodeY, **gpArguments) 
            
            self.model.setGP(nodeX, nodeY, likelihood =  5e-2* nodeY.variance,lengthscales = self.Priorlengthscale)
            self.model.color = np.asarray(color)
            self.model.color2 = np.asarray(color2)
            self.model.niter = 0
            self.model.Initiallengthscales = self.Priorlengthscale
            self.model.omega = omega_2_prior
            self.model.title = title
            self.model.directory = directory
            
    def fit(self, steps = training.getTrainStrategyAlternate(1), initialise = 'PCA',save = False,reconstruction_error = False):
        t = time()
        if initialise == 'PCA':
            initialisation.initialiseLatentVariablesPCA(self.model)
        elif initialise == 'Diffmap':
            initialisation.initialiseLatentVariableDM(self.model)
        elif initialise == 'random':
            self.initialised = True
            initialisation.initialiseLatentVariablesRandom(self.model)
        elif isinstance(initialise, np.ndarray):
            self.model.nodes['X'].setValues(initialise)
            self.model.initialised = True
        else:
            raise ValueError()
        print('Initialisation : '+str(time()-t)+'s')
        # Initialise and restore latent variables?
        print('Training...')
        self.model.initialisation = self.x
        self.model.train('sequence', steps = steps,save = save,reconstruction_error=reconstruction_error)
                                                  
    @property
    def x(self):
        if self.cleanNaN:
            x = np.nan * np.zeros((self.nSamples, self.model.nodes['X'].value.shape[1]))
            x[self.idxNotnan] = self.model.nodes['X'].value
            return x
        else:
            return self.model.nodes['X'].value

    def transform(self, Y):
        Y = self.dataTransformer.transform(Y)
        x = tf.Variable(x0, dtype = tf.float64)
        def loss(h = h):
            f_mean, f_var = self.getGP('X', 'Y').predict_f(x)
            lossLikelihood = tf.linalg.norm(Y -  f_mean)
            lossPrior = tf.reduce_sum(h.model.nodes['X'].prior.log_prob(x))
            return lossLikelihood

        opt = gpflow.optimizers.Scipy()
        maxiter = ci_niter(1000)
        res = opt.minimize(
            loss, 
            method='L-BFGS-B',
            variables=  [x],
            compile = False,
            options=dict(maxiter=maxiter),
        ) 
        return x.numpy()
        
    def inverse_transform(self, X):
        f_mean, f_var = self.model.getGP('X', 'Y').predict_f(X)
        return f_mean