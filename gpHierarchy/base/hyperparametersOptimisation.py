import tensorflow as tf
import collections, numpy as np
import time, sklearn
from bayes_opt import BayesianOptimization
import abc

def reconstructionPredictor(X, Y, model, k = 10):
    cv = sklearn.model_selection.KFold(k)
    Ypred = np.zeros(Y.shape)
    idTest = []
    for idtrain, idtest in cv.split(X):
        model.fit(X[idtrain], Y[idtrain])
        Ypred[idtest] = model.predict(X[idtest]).reshape(Ypred[idtest].shape)
    return Ypred

def reconstructionError(X, Y, model, k = 10):
    Ypred = reconstructionPredictor(X, Y, model, k)
    return np.mean(np.mean(np.abs(Ypred - Y), axis = 0))


def getHyperparameterListFromModel(model):
    # For each GP
    for gp in model.gps:
        pass
    # For each prior
    pass


Hyperparameter = collections.namedtuple('Hyperparameter', 'name set get minVal maxVal ')
class HyperparameterList:
    """
    
    """
    def __init__(self, model, nodeX, Y_prediction):
        #self.
        self.hyperparams = collections.OrderedDict()
        self.nodeX =  nodeX
        self.Y_prediction = Y_prediction
        self.model = model
        self.saveModelState()
        
    def addParameter(self, var, minVal, maxVal):
        for i in range(var.length):
            name = '%s_%d' % (var.name, i)
            self.hyperparams[name] = Hyperparameter(name, lambda theta, i = i: var.modify(theta, i),
                                                            lambda i = i: var.getRaw(i),
                                                           var.transform.inverse_transform(minVal[i]),
                                                        var.transform.inverse_transform(maxVal[i]))
            
    def resetModelState(self):
        self.model.loadLatentVariables(self.modelState)
        self.setValues(**self.savedHyperparameters)
        
    def saveModelState(self):
        self.modelState = self.model.getLatentVariablesValues()
        self.savedHyperparameters = self.getValues()
        
    def setValues(self, **params):
        """
        Apply the variables
        """
        for n,xx in params.items():
            self.hyperparams[n].set(xx)
            
    def getValues(self):
        """
        Apply the variables
        """
        r = collections.OrderedDict()
        for n in self.hyperparams:
            r[n] = self.hyperparams[n].get()
        return r
    
    def getCVLoss(self,*theta_0,  **theta_1):
        
        if len(theta_0):
            theta_0 = theta_0[0]
            self.setValues(**{n : theta_0[i] for i,n in enumerate(self.hyperparams)})
        else:
            self.setValues(**theta_1)
            
        #Train model
        self.model.loadLatentVariables(self.modelState)
        self.model.train() 
        
        # Compute loss
        # Not sure if overfitting in the embedding. What is the right way of assess an embedding?
        idx = self.Y_prediction[:, 0] == self.Y_prediction[:, 0]
        return reconstructionError(self.nodeX.value[idx], self.Y_prediction[idx], k = 10)
    
    def getVariablesForOptimisation(self):
        r = {}
        for p in self.hyperparams.values():
            r[p.name] = (p.minVal, p.maxVal)
        return r
    
    def optimise(self, init_points = 20, n_iter = 100):
        self.optimizer = BayesianOptimization(
            f=lambda **theta: - self.getCVLoss(**theta),
            pbounds=self.getVariablesForOptimisation(),
            random_state=1,
        )

        self.optimizer.maximize(
            init_points=init_points,
            n_iter=n_iter,
        )
        return self.optimizer
    
class Transform:
    def __call__(self, x):
        return self.transform(x)


class TransformExponential(Transform):
    def transform(self, X):
        return np.exp(X)
    
    def inverse_transform(self, X):
        return np.log(X)
    
class TransformIdentity(Transform):
    def transform(self, X):
        return X
    
    def inverse_transform(self, X):
        return X
    
class TransformLinear(Transform):
    def __init__(self, b = 1, bias = 0):
        self.b = b
        self.bias = bias
    def transform(self, X):
        return X*self.b + self.bias
    
    def inverse_transform(self, X):
        return (X - self.bias)/self.b

class Variable(abc.ABC):
    def __init__(self, var,  name = None, transform = TransformIdentity()):
        self.var = var
        self.name = name
        self.transform = transform
        
    def getRaw(self, i):
        return self.transform.inverse_transform(self.value[i])
    
    @property
    def length(self):
        return len(self.value)
    
    @abc.abstractmethod
    def modify(self, val, i):
        pass
    
    
class TensorflowVariableModifier(Variable):
    @property
    def value(self):
        return self.var.numpy() 
    
    def getHyperparameterList(self, rangeMax = 1e-3, rangeMin=1e3):
        pass
    
    def getBounds(self, i):
        return

class TensorflowVectorVariableModifier(TensorflowVariableModifier):
    """
    Wrapper for modifying the values
    """
        
    def modify(self, val, i):
        value = self.value
        value[i] = self.transform(val)
        self.var.assign(value)
    
class TensorflowScalarVariableModifier(TensorflowVariableModifier):
    """
    Wrapper for modifying the values
    """
        
    def modify(self, val, i):
        if i != 0:
            raise ValueError()
        self.var.assign(self.transform(val))
    
    def getRaw(self, i):
        return self.transform.inverse_transform(self.value)

    @property
    def length(self):
        return 1