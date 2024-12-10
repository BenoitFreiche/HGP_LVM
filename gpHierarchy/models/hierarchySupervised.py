import numpy as np, functools
from . import GPC, weightModel
import sklearn

def reweight(w, y, k):
    """
    Reweight so that positive samples have a prior probability of k, 
    """
    k_0 = 1
    k_1 = k *k_0/(1 - k)
    
    w_0 = np.sum(w[y == 0.])
    w_1 = np.sum(w[y == 1.])

    w = w.copy()
    w[y == 0.] *= k_0/w_0
    w[y == 1.] *= k_1/w_1
    return w


logit = lambda z: np.log(z/(1 - z))
logistic = lambda z: 1/(1 + np.exp(-z))
def shiftProbs(z, t):
    return logistic(logit(z) + t)

def shiftPredictionProbability(func, t ):
    @functools.wraps(func)
    def f(*args, **kwargs):
        y = func(*args, **kwargs)
        y[:,1] = shiftProbs(y[:, 1], t)
        y[:,0] = 1 - y[:, 1]
        return y
    return f


class HierarchyThreshold:
    def __init__(self,  threshold = 0.1):
        self.modelX0 = GPC.GPC()
        self.modelX1 = GPC.GPC()
        self.threshold = threshold

    def fit(self, X_0, X_1, y):
        w = np.ones(X_0.shape[0])/2
        modelX0 = self.modelX0.fit(X_0, y)
        modelX1 = self.modelX1.fit(X_1, y)
            
            
    def predict_proba(self, X_0, X_1):
        y0 = self.modelX0.predict_proba(X_0)[:, 1]
        y1 = self.modelX1.predict_proba(X_1)[:, 1]
        i0 = np.abs(y0 - 0.5) < self.threshold
        return y0 *i0 + (1 - i0) * y1
    
    def predict_proba_test(self, X_0, X_1):
        y0 = self.modelX0.predict_proba(X_0)[:, 1]
        y1 = self.modelX1.predict_proba(X_1)[:, 1]
        i0 = np.abs(y0 - 0.5) < self.threshold
        return y0 *i0 + (1 - i0) * y1,  i0

    def crossval_predict(self, X_0, X_1,  Y, k = 10):
        cv = sklearn.model_selection.StratifiedKFold(n_splits=k)
        res = np.zeros(Y.shape)
        w = np.zeros(Y.shape[0])
        idTest = []
        for idtrain, idtest in cv.split(X_0, Y ):
            self.fit(X_0[idtrain], X_1[idtrain], Y[idtrain])
            res[idtest], w[idtest] = self.predict_proba_test(X_0[idtest], X_1[idtest])
        return res, w

#  Optimise:
#  $$
#  Pr( Y | X, W) \cdot Pr (W)
#  $$
#  Where X is the data, and W the level which we explore
#  $$
#  Pr(Y | X, W) = Pr( W) * Pr ( Y| X_0) + (1 - Pr(W) * Pr ( Y | X_1, W)
#  $$
#  
#  We maximise the log-likelihood, using Jensen's inequality. For the moment, we use an SVC classifier since we can add sample weights very easily. Next step do it for gpflow.
#  Since the logarithm is monotonic, concave:
#  $$
#  log(a/2 + b/2) \le log(a)/2 + log(b)/2
#  $$
#  
#  Or equivalently:
#  $$
#  log(a + b) \le log(a)/2 + log(b)/2 - 2 log(1/2)
#  $$
#  
#  We can apply that property to the likelihod
#  $$
#  log (Pr(W|X_0)(Pr(Y|X_0) - Pr(Y|X_1) + Pr(Y|X_1)) \le log (Pr(W|X_0)  log (Pr(Y|X_0) - Pr(Y|X_1))  + (1 - Pr(Y|X_0) + Pr(Y|X_1)))
#  $$
    

class HierarchySupervisedPredictErrorImprovement:
    def __init__(self, nItsTrain = 3, shiftProbsLeveltransition = 0, threshold = 0.5, model = GPC.GPC, argsModelW = {}, logWeights = True):
        self.modelX0 = model()
        self.modelX1 = model()
        
        self.nIts = nItsTrain
        self.shiftProbs = shiftProbsLeveltransition
        self.threshold = threshold
        
        self.argsModelW = argsModelW
        self.logWeights = True
    def fit(self, X_0, X_1, y, w0 = None):
        try:
            self.modelX0.prepareModel(X_0)
            self.modelX1.prepareModel(X_1)
        except:
            pass
        if w0 is None:
            w0 = np.ones(X_0.shape[0])/2
        modelX0 = self.modelX0
        modelX1 = self.modelX1
        self.modelW  = weightModel.SampleWeightModel(X_0, w0, **self.argsModelW)
        if self.shiftProbs:
            self.modelW.predict_proba = shiftPredictionProbability(self.modelW.predict_proba, self.shiftProbs)

        X_0 = np.array(X_0, order = 'C')
        X_1 = np.array(X_1,  order = 'C')
        y = np.array(y,  order = 'C')
        w = self.modelW.predict_proba(X_0)[:, 1]
        # TODO: add convergence automatic test
        for i in range(self.nIts):
            # Train the other models
            modelX0.fit(X_0, y, w)
            modelX1.fit(X_1, y, 1 - w)
            
            y_pred_0 = modelX0.predict_proba(X_0)[:, 1]
            y_pred_1 = modelX1.predict_proba(X_1)[:, 1]

            z_0 = np.log(1 - np.abs(y - y_pred_0))
            z_1 = np.log(1 - np.abs(y - y_pred_1))
            if self.logWeights:
                sampleWeights_W = z_0 - z_1
            else:
                sampleWeights_W = (1 - np.abs(y - y_pred_0)) - (1 -np.abs(y - y_pred_1))
            self.modelW.fit(X_0, sampleWeights_W)

            w = self.modelW.predict_proba(X_0)[:, 1]

    def predict_proba(self, X_0, X_1, hard = True):
        modelX0 = self.modelX0
        modelX1 = self.modelX1
        modelW = self.modelW    
        
        w = modelW.predict_proba(X_0)[:, 1]
        if hard:
            w = w > self.threshold
        elif False:
            w = w > self.threshold + (w <= self.threshold) * w
            
        prob_X0 = modelX0.predict_proba(X_0)[:, 1]
        prob_X1 = modelX1.predict_proba(X_1)[:, 1]
        
        prob_0 = w *(1- prob_X0)  + (1- w) * (1 - prob_X1)
        prob_1 = w * prob_X0  + (1 - w) * prob_X1
        return prob_1
    
    
    def crossval_predict(self, X_0, X_1,  Y, k = 10):
        if isinstance(k, int):
            cv = sklearn.model_selection.StratifiedKFold(n_splits=k)
        else:
            cv = k
        res = np.zeros(Y.shape)
        w = np.zeros(Y.shape[0])
        idTest = []
        for idtrain, idtest in cv.split(X_0, Y ):
            self.fit(X_0[idtrain], X_1[idtrain], Y[idtrain])
            res[idtest], w[idtest] = self.predict_proba_test(X_0[idtest], X_1[idtest])
        return res, w
    
    def crossval_predict_debug(self, X_0, X_1,  Y, k = 10):
        if isinstance(k, int):
            cv = sklearn.model_selection.StratifiedKFold(n_splits=k)
        else:
            cv = k
        res1 = np.zeros(Y.shape)
        res2= np.zeros(Y.shape)

        w = np.zeros(Y.shape[0])
        idTest = []
        for idtrain, idtest in cv.split(X_0, Y ):
            self.fit(X_0[idtrain], X_1[idtrain], Y[idtrain])
            res1[idtest] = self.modelX0.predict_proba(X_0[idtest])[:,1]
            res2[idtest] = self.modelX1.predict_proba(X_1[idtest])[:,1]
            w[idtest] = self.modelW.predict_proba(X_0[idtest])[:,1]
        return res1, res2, w
