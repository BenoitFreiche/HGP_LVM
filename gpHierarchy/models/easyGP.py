import scipy, scipy.interpolate
import numpy as np, itertools, collections, logging

import sklearn
from sklearn.impute import SimpleImputer
import GPy
from scipy.interpolate import interp1d

 
class EasyGP:
    """
    Wrapper for the GP class of GPy (is a bit easier than gpflow, but to avoid redundant classes it would be better to use gpflow)
    """
    def __init__(self, Xdim, mode = 'regression'):
        self.ndimsInput = Xdim
        self.ker = GPy.kern.ExpQuad(Xdim, ARD=1)  + GPy.kern.White(Xdim)
        self.mode = mode
        if mode == 'regression':
            self.ker += GPy.kern.Bias(Xdim)
        
    def fit(self, X, y):
        if len(y.shape) == 1:
            y = y.reshape((-1, 1))

        idx = np.logical_and(X[:,0] == X[:,0], y[:,0] == y[:, 0])
        X = X[idx]
        y = y[idx]
        if self.mode == 'regression':
            self.y_mean = np.mean(y, axis = 0)
            y = y - self.y_mean
            self.m = GPy.models.GPRegression(X, y, kernel = self.ker)
        elif self.mode == 'classification':
            self.m = GPy.models.GPClassification(X, y, kernel = self.ker)
        elif self.mode == 'streamline':
            self.m = GPy.models.GPRegression(X, y)
        else:
            raise ValueError()
        self.m.optimize(max_f_eval = 5000)
        self.X = X
    
    def predict(self, X):
        Y_pred = self.m.predict(X)[0] 
        if self.mode == 'regression':
            Y_pred += self.y_mean
        return Y_pred #Get only the mean, not the variance
   
    def score(self, X, y):
        ypred = self.predict(X)
        return self.m.score(x, y)
    
    def crossval_predict(self,X, Y, k = 10 ):
        """
        Warning, only tested for 1 dimensional output
        """
        if self.mode == 'classification':
            cv = sklearn.model_selection.StratifiedKFold(k)
        else:
            cv = sklearn.model_selection.KFold(k)
        res = np.zeros(Y.shape)
        idTest = []
        for idtrain, idtest in cv.split(X, Y ):
            self.__init__(self.ndimsInput, self.mode)
            self.fit(X[idtrain], Y[idtrain])
            res[idtest] = self.predict(X[idtest]).reshape(res[idtest].shape)
        return res       
    
    def streamline(self, x0, incrementX = 0.5,  epsIncrement = 1e-2, epsGradient = 1e-4, 
                   increasing = True, maxPointsComputation = 10000, pointsStreamline = 100, normalise = 'embedding'):
        x = x0.copy()
        incX = 0
        eps = epsIncrement
        k = 1 if increasing else -1
        grad_x = np.ones(x0.shape) * (epsGradient+ 1)
        allXs = [x.copy()]
        
        if self.mode == 'classification':
            logging.warning('We cannot obtain the jacobian using a classification model, obtaining probabilities and train using a surrogate model')
            model = EasyGP(self.ndimsInput)
            model.fit(self.X, self.predict(self.X))
            model = model.m
        else:
            model = self.m
        
        while incX < incrementX and np.linalg.norm(grad_x) >epsGradient and len(allXs) < maxPointsComputation:
            grad_x, _ = model.predict_jacobian(x.reshape((1, -1)))
            incX += eps * np.linalg.norm(grad_x)
            x  += k * eps * grad_x.flatten()
            allXs.append(x.copy())
        xs = np.array(allXs)
        if normalise == 'embedding':
            d = np.pad(np.cumsum(np.linalg.norm(xs[1:, :] - xs[-1:, :], axis = 1)), (1, 0), mode = 'constant')
            
        elif normalise == 'output':
            d = self.predict(xs).flatten()
        
        elif normalise == 'streamline':
            d = np.linspace(0,1, num= len(xs))
        else:
            raise ValueError()
        newX = np.linspace(d[0], d[-1], pointsStreamline)
        xs = interp1d(d, xs, axis = 0)(newX)

        return np.array(xs)
    
def getContours2D(X, y, model = None):
    if model is None:
        mode = 'regression' if len(np.unique(y)) > 3 else 'classification'
        model = EasyGP(2, mode = mode)
        
        model.fit(X, y)

    XX, YY = np.meshgrid(np.linspace(np.min(X[:,0]), np.max(X[:,0])),
                         np.linspace(np.min(X[:,1]), np.max(X[:,1])))
    XY = np.stack([XX, YY], axis =2).T.reshape((2, -1)).T
    Z = model.predict(XY)
    ZZ = Z.reshape(XX.shape).T
    return XX, YY, ZZ
