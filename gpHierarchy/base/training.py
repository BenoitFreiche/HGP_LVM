from . import hierarchyGraph
import gpflow as gp, tensorflow as tf
# from gpflow.ci_utils import ci_niter
import numpy as np
from gpHierarchy.models import mskr
import matplotlib.pyplot as plt
import gpflow.logdensities, gpflow.base
import logging
import os
from time import time
import pickle
import sys

def trainSequence(self, steps,save = False,reconstruction_error = False):
    for s in steps:    
        if 'updateX0' in s:
            self.storeInitialData()
        else:
            trainBase(self,save = save,reconstruction_error = reconstruction_error, **s)

def callback(xk):
    print(xk)
#     return s
    
def trainBase(self, maxiter = 1000, debug = True, trainHyperparams = False, trainEmbedding = True, 
          extraTrainingVars = [], trainCallback = None, method = 'L-BFGS-B',iprint = 101, lambda_X0 = 0, compile = False,
          save = False,reconstruction_error = False):
    # method = 'L-BFGS-B',
    
    if hasattr(self, 'loss_values'):
        self.current_iter = 0
    Nfeval = 1

    if not self.initialised:
        raise ValueError('Latent variables were never initialised')

    logging.debug(f'Loss before optimisation {self.loss()}')
    

        
    if trainHyperparams:
        print('TRAINING HYPERPARAMS....')

        hyperparams =  [v  for gp in self.gpLinkFunctions.values() for v in gp.trainable_variables[:] if len(v.shape) <= 1] + extraTrainingVars 
        for n in self.latentVariables.values():
            hyperparams += list(n.prior.trainable_variables)

    else:
        print('TRAINING EMBEDDING....')
        hyperparams = []
    
    embeddingVariables =  self.getLatentVariables(trainable = True) if trainEmbedding else []
    
    oldX = self.getLatentVariablesValues()
    n_embeddings = len(oldX)
    
    t=time()

    if method == 'Adam':
        opt = tf.optimizers.Adam(learning_rate = 1)
        for _ in range(maxiter):
            self.resultTrain = opt.minimize(
                self.loss if lambda_X0 == 0 else lambda : self.loss() + lambda_X0 * self.loss_X0(), 
                 embeddingVariables + hyperparams
            )

    else:
        opt = gp.optimizers.Scipy()
        self.resultTrain = opt.minimize(
            self.loss if lambda_X0 == 0 else lambda : self.loss() + lambda_X0 * self.loss_X0(), 
            variables =  embeddingVariables + hyperparams,
            compile = compile,
            callback = trainCallback, # store something 
            # options=dict(iprint = 1),
            options=dict(maxiter=1,iprint = 1),
            method = method,
        )       
    print('Optim : '+ str(np.round(time()-t,2))+' s')
    logging.debug(f'Loss after optimisation {self.loss()}')
    
    
def getTrainStrategyAlternate(nIts, **kwargs):
    l = [ 'updateX0']
    for i in range(nIts):
        l.append({'trainHyperparams' : True, 'trainEmbedding' : False})
        l.append({'trainHyperparams' : False, 'trainEmbedding' : True, 'lambda_X0' : kwargs.get('lambda_X0', 0)})
    return l

def getTrainStrategySimultane(nIts, **kwargs):
    l = [ 'updateX0']
    for i in range(nIts):
        l.append({'trainHyperparams' : True, 'trainEmbedding' : True})
    return l

def getTrainStrategyDecreasingProximal(nIts, lambda_X0_begin, lambda_X0_end):
    l = [ 'updateX0']
    for lambda_x0i in np.logspace(np.log10(lambda_X0_begin), np.log10(lambda_X0_end), num = nIts):
        l.append({'trainHyperparams' : True, 'trainEmbedding' : False})
        l.append({'trainHyperparams' : False, 'trainEmbedding' : True, 'lambda_X0' : lambda_x0i})
    return l

def train(self, mode = 'base',save = False,reconstruction_error = False, **kwargs):
    if mode == 'base':
        return trainBase(self,save = save,reconstruction_error = reconstruction_error,**kwargs)
    elif mode == 'sequence':
        return trainSequence(self,save = save,reconstruction_error = reconstruction_error,**kwargs)
    else:
        raise ValueError('Possible values {base, sequence}')
        

setattr(hierarchyGraph.Graph, 'train', train)

