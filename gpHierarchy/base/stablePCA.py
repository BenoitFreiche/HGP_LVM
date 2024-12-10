"""
PCA where I solve the sign undetermination by imposing that the first nonzero component of each embedding has to be positive.

It also works with missing data (nan)
"""

import sklearn.decomposition, numpy as np
def stablePCAmodes(X):
    X = X.T
    idx = ( np.abs(X)  > 1e-6).argmax(axis=0)
    signs = np.sign(X[(idx, range(len(idx)))])
    return (X * signs).T

def stableEmbeddings(X):
    idx = ( np.abs(X)  > 1e-6).argmax(axis=0)
    signs = np.sign(X[(idx, range(len(idx)))])
    return X * signs

class StablePCA(sklearn.decomposition.PCA):
        
    def fit(self, X, y = None):
        idx = np.all(X == X, axis = 1)

        sklearn.decomposition.PCA.fit(self, X[idx])
        self.components_ = stablePCAmodes(self.components_)
        
    def transform(self, X, y = None):
        idx = np.all(X == X, axis = 1)
        Y = np.nan * np.zeros((len(X), self.components_.shape[0]))
        Y[idx] = sklearn.decomposition.PCA.transform(self, X[idx])
        self.components_ = stablePCAmodes(self.components_)
        return Y
    
    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
