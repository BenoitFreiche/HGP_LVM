import tensorflow as tf, gpflow,  tensorflow_probability as tfp

from gpflow.utilities import ops, print_summary
from gpflow.config import set_default_float, default_float, set_default_summary_fmt
# from gpflow.ci_utils import ci_niter
import numpy as np

import gpflow.logdensities, gpflow.base
from typing import Optional, Tuple
import abc, sklearn, pathlib, logging, collections
from . import gpflowExtensions
# logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
#     datefmt='%Y-%m-%d:%H:%M:%S',
#     level=logging.DEBUG)
globalModel = None

def getGlobalModel():
    return globalModel


"""
Prior

TODO: CHeck with Gabriel The GaussianPrior
"""

def getGaussianPrior(nDims, omega_2 = 1):
    if omega_2 > 0:
        return PriorGaussian(nDims, omega_2)
    else:
        return NoPrior()


class PriorGaussian(gpflow.base.Module):
    "https://gpflow.github.io/GPflow/2.4.0/_modules/gpflow/base.html#Module"
    def __init__ (self, nDims, scale = 1, mu = None):
        self.normalUnit = tfp.distributions.MultivariateNormalDiag(loc=np.zeros(nDims), scale_diag = np.ones(nDims))
        self.set_omega_2(scale)
        self.mu = mu if mu is not None else np.zeros(nDims)
        
    def set_omega_2(self, val):
        self.omega2 = val
        self.omega = gpflow.Parameter(np.log(np.sqrt(val)))
        
    @tf.function
    def log_prob(self, X):
        #X = X[X[0] == X[0]]
        return self.normalUnit.log_prob((X - self.mu)/tf.exp(self.omega))   
    
    @classmethod
    def getGaussianPrior(cls, nDims, omega_2 = 1):
        return cls(nDims, omega_2)
    
class NoPrior(gpflow.base.Module):
    def log_prob (self, X):
        return tf.constant(0, X.dtype)

class Link:
    #Unused
    def __init__(self, n1, n2, kernel = 'rbf', model = None, **kwargs):
        if model is None:
            model = globalModel
        if not isinstance(model, Graph):
            raise ValueError()
        model.addLink(n1, n2, kernel = kernel, **kwargs)
        
class Node(abc.ABC):
    def __init__(self, **kwargs):
        if kwargs.get('name', None):
            self.name = kwargs.get('name')
        model = kwargs.get('model', globalModel)
        if model:
            model.addNode(self)

    def getNonMissingData(self):
        
    # =============================================================================
    #                MODIF HERE 
    # =============================================================================
        r = np.all(np.asarray(self.value) == np.asarray(self.value), axis = 1)
        if len(r.shape) != 1:
            r = r[:,0]
        return r
    @abc.abstractmethod
    def value(self):
        raise NotImplemented()
        
    @abc.abstractmethod
    def copy(self):
        raise NotImplemented()

    def dim(self):
        return self.X.shape[1]
    
    @tf.function
    def prior_loss(self):
        return 0.
    
class ObservedVariable(Node):
    def __init__(self, X, varianceType = 1, **kwargs):
        self.X =  tf.Variable(X)
        if len(self.X.shape) == 1:
            self.X = self.X.reshape((-1, 1))
        if kwargs.get('removeMean', True):
            self.mean = np.nanmean(self.X.numpy(), axis = 0).reshape((1, -1))

            self.X.assign(self.X - np.nanmean(self.X.numpy(), axis = 0).reshape((1, -1)))
        else:
            self.mean = np.zeros(self.X.shape[1])
            
        if kwargs.get('normalizeSTD', False):
            self.X.assign(self.X /np.nanstd(self.X.numpy(), axis = 0).reshape((1, -1)))

        if varianceType == 'automatic':
            self.variance = np.nanmedian(np.nanvar(self.value, axis = 0))
            if self.variance == 0:
                print('Warning : median.variance = 0, replacing by mean')
                self.variance = np.nanmean(np.nanvar(self.value, axis = 0))
        else:
            self.variance = varianceType
        super().__init__(**kwargs)

    @property
    def value(self):
        return self.X.numpy()
    
    def setValue(self, X):
        self.X.assign(X)
    
    def copy(self):
        n = ObservedVariable(self.X, removeMean = False, normalizeSTD = False, varianceType = self.variance, name = self.name)
        return n
        
class LatentVariable(Node):
    def __init__(self, X = None, size = None, prior = None, **kwargs):
        if X is not None:
            self.X =  tf.Variable(X)
            self.initialised = True
        elif size is not None:
            self.X  =  tf.Variable(np.zeros(size))
            self.initialised = False
        else:
            raise ValueError()
            
        self.variance = 1
        if prior is None or not prior:
            self.prior = NoPrior()
        elif isinstance(prior, float) or isinstance(prior, int):
            self.prior = getGaussianPrior(self.X.shape[1], prior)
        elif prior is not None:
            self.prior = prior
        else:
            self.prior = NoPrior()
        self.priorityLevel = kwargs.pop('priorityLevel', 0)
        self.trainable = True
        super().__init__(**kwargs)
    
    @property
    def value(self):
        return self.X.numpy()

    def estimatedVariance(self):
        return 1
    
    def storeInitialData(self):
        self.X0 = self.value
        
    def loss_X0(self):
        return tf.linalg.norm(self.X - self.X0)**2
    
    @tf.function
    def prior_loss(self):
        if self.prior is None:
            return tf.constant(0., self.X.dtype)
        else:
            return - tf.reduce_sum(self.prior.log_prob(self.X))
    def setValues(self, X):
        self.X.assign(X)
        
    def copy(self):
        n = LatentVariable(self.value, prior = self.prior, name = self.name)
        return n


class Graph:
    def __init__(self, names = None):
        self.nodes = {}
        self.nodesByPriority = collections.defaultdict(set)
        self.latentVariables = {}
        self.adjacency = collections.defaultdict(set)
        self.inverseAdjacency = collections.defaultdict(set)
        self.gpLinkFunctions = {}
        self. names = names
        self.initialised = False

    def addNode(self, n):
        if n.name in self.nodes:
            raise ValueError('Repeated node name')
        self.nodes[n.name] = n
        if isinstance(n, LatentVariable):
            self.latentVariables[n.name] = n
            self.nodesByPriority[n.priorityLevel].add(n.name)
    
    def addLink(self, n1, n2, kernel = 'rbf', **kwargs):
        """
        Add directed link between two variables.
        
        TODO: clean
        """
        if isinstance(n1, str):
            n1 = self.nodes[n1]
        if isinstance(n2, str):
            n2 = self.nodes[n2]

        if n1.name not in self.nodes or n2.name  not in self.nodes:
            raise ValueError('Trying to add a link between nodes that do not exist in the graph!')
            
        #TODO: vérifier ce qu'il y a derrière ces adjacency 
        self.adjacency[n1.name].add(n2.name)
        self.inverseAdjacency[n2.name].add(n1.name)

        #Create GP
        var = n2.variance
        nDimsInput = n1.dim() +  (kwargs.get('fixedData').shape[1] if kwargs.get('gp_type', None) == 'fixedDataRegression' else 0)
        
        
        # =============================================================================
        #TODO:  Problèmes ici, on ne peut pas gérer
        # la lengthscales et on ne l'entraine pas parce que ça bug
        #DONE
        # =============================================================================
        
        
        if kernel == 'rbf':
            if kwargs.get('lengthscales'):
                ls = kwargs.get('lengthscales')
                kernel = gpflow.kernels.SquaredExponential(lengthscales=ls, variance = var)
            else:
                kernel = gpflow.kernels.SquaredExponential(lengthscales=tf.ones((nDimsInput,)), variance = var)
        elif kernel == 'linear':
            kernel = gpflow.kernels.Linear(variance = var * tf.ones((nDimsInput,)))
        else:
            kernel = kernel
            
        # Add data
        data = (n1.X, n2.X)
        
        # Type of kernel
        if kwargs.get('gp_type', 'regression') == 'regression':
            n2.getNonMissingData()
            gp = gpflowExtensions.GPRMissingData(data, np.where(np.logical_and(n1.getNonMissingData(), n2.getNonMissingData()))[0], kernel = kernel)
        elif kwargs.get('gp_type') == 'kernel_trick':
            gp = gpflowExtensions.GPRKernel(data, np.where(np.logical_and(n1.getNonMissingData(), n2.getNonMissingData()))[0], kernel = kernel)
        elif kwargs.get('gp_type') == 'fixedDataRegression':
            gp = gpflowExtensions.GPRFixedData(kwargs.get('fixedData'), data, np.where(np.logical_and(n1.getNonMissingData(), n2.getNonMissingData()))[0], kernel = kernel)
        else:
            raise ValueError()
        gp.weight = kwargs.get('weight', 1)
        
        #If n2 is observed, set variance to the observed one.
        self.gpLinkFunctions[(n1.name, n2.name)] = gp
        
        
        # =============================================================================
        #  TODO:       A verifier et à modifier
        # =============================================================================
        # Add prior to the lengthscales, if the option is set
        if 'priorLengthscales' in kwargs:
            gp.kernel.lengthscales.prior = kwargs.get('priorLengthscales', None) 
            # tfp.distributions.Gamma(concentration = 0.75* np.ones(n1.dim()), rate = np.ones(n1.dim()))  


    def loss(self):
        if hasattr(self,'loss_values'):
            try:
                for i,n in enumerate(self.gpLinkFunctions.values()):
                    self.loss_values['gp{0}'.format(i)].append(tf.reduce_sum(n.training_loss()).numpy())
        
                for i,n in enumerate(self.latentVariables.values()):
                    self.loss_values['prior{0}'.format(i)].append(tf.reduce_sum(n.prior_loss()).numpy())
            except:
                for n in self.gpLinkFunctions.values():
                    self.loss_values['gp'].append(tf.reduce_sum(n.training_loss()).numpy())
        
                for n in self.latentVariables.values():
                    self.loss_values['prior'].append(tf.reduce_sum(n.prior_loss()).numpy())
            self.current_iter+=1
            if 'x' in self.loss_values.keys():
                self.loss_values['x'].append(self.latentVariables['X_1'].X.numpy())
        return tf.reduce_sum([gp.training_loss() for gp in self.gpLinkFunctions.values()]) + tf.reduce_sum([n.prior_loss() for n in self.latentVariables.values()])
    
    @tf.function
    def loss_X0(self):
        """
        Virtual loss that penalises going far from the initial solution
        """
        return tf.reduce_sum([n.loss_X0() for n in self.latentVariables.values()])

    def resetLatentVariables(self):
        for n in self.nodes.values():
            if isinstance(n, LatentVariable):
                n.initialised = False
            
    def getLatentVariables(self, trainable = False):
        return [l.X for l in self.latentVariables.values() if not trainable or l.trainable]
    
    def getGP(self, n1, n2):
        try:
            n1 = n1.name
        except:
            pass
        try:
            n2 = n2.name
        except:
            pass
        return self.gpLinkFunctions[(n1, n2)] 
            
    def setGP(self, n1, n2, **kwargs):
        gp = self.getGP(n1, n2)
        for k, val in kwargs.items():
            if k == 'variance':
                gp.kernel.variance.assign(val)
            elif k == 'likelihood':
                gp.likelihood.variance.assign(val)
            elif k == 'lengthscales':
                gp.kernel.lengthscales.assign(val)
            elif k == 'weight':
                pass
            else:
                raise ValueError(f'Unknown parameter {k}')
    # Could use a template / meta
    def deepcopy(self):
        m = Graph()
        for n, node in  self.nodes.items():
            newNode = node.copy()
            m.addNode(newNode)
    
        for (n1, n2) in self.gpLinkFunctions:
            m.addLink(n1, n2)
            gp1 = self.getGP(n1, n2)
            gp2 = m.getGP(n1, n2)
            gp2.kernel = gpflow.utilities.deepcopy(gp1.kernel)
            gp2.likelihood =  gpflow.utilities.deepcopy(gp1.likelihood)
        return m
    
    
    
    def loadLatentVariables(self, data):
        if isinstance(data, str) or isinstance(data, pathlib.Path):
            data = np.load(data)
            
        for i, X in data.items():
            if i in self.latentVariables:
                self.latentVariables[i].setValues(X)
                
    def getLatentVariablesValues(self):
        return {str(i) : n.X.numpy() for i,n in self.latentVariables.items()}    

    def saveLatentVariables(self, path):
        np.savez(path, **self.getLatentVariablesValues(), names = self.names)
        
    def storeInitialData(self):
        for n in self.latentVariables.values():
            n.storeInitialData()

                        
    # To make less verbouse model setting,for use of  with model
    def __enter__ (self):
        global globalModel
        self.oldModel = globalModel
        globalModel = self
    def __exit__ (self, type, value, tb):
        global globalModel
        globalModel = self.oldModel
