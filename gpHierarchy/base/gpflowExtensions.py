import numpy as np, itertools, collections, logging, sklearn

import tensorflow as tf
import gpflow as gp
from gpflow.utilities import set_trainable
from gpflow.config import set_default_float, default_float, set_default_summary_fmt
# from gpflow.ci_utils import ci_niter
import gpflow.logdensities
from typing import Optional, Tuple

        
class GPRMissingData(gpflow.models.GPR):
    """
    Wrapper to make it possible to work with nans
    """
    def __init__(self,
        data: gpflow.models.training_mixins.RegressionData,
        idx : tf.Tensor,
        kernel: gpflow.kernels.Kernel,
        mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
        noise_variance: float = 1.0,
        weight: float = 1.
    ):
        self.idx = idx
        self.weight = weight
        super().__init__(data, kernel, mean_function, noise_variance)
        try:
            set_trainable(self.kernel.lengthscales, True)
        except:
            print('non trainable lengthscale')
            pass
    def log_marginal_likelihood(self, X =None, Y = None) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        if X is None:
            X, Y = self.data
            X = tf.gather(X, self.idx, axis = 0)
            Y = tf.gather(Y, self.idx, axis = 0)
        K = self.kernel(X)
        num_data = tf.shape(X)[0]
        k_diag = tf.linalg.diag_part(K)
        s_diag = tf.fill([num_data], self.likelihood.variance)
        ks = tf.linalg.set_diag(K, k_diag + s_diag)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = gpflow.logdensities.multivariate_normal(Y, m, L)
        return tf.math.scalar_mul(self.weight, tf.reduce_sum(log_prob))

    def _add_noise_cov(self, K: tf.Tensor) -> tf.Tensor:
        """
        Returns K + σ² I, where σ² is the likelihood noise variance (scalar),
        and I is the corresponding identity matrix.
        """
        k_diag = tf.linalg.diag_part(K)
        s_diag = tf.fill(tf.shape(k_diag), self.likelihood.variance)
        return tf.linalg.set_diag(K, k_diag + s_diag)
    def predict_f(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ):
        r"""
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        X_data, Y_data = self.data
        X_data = tf.gather(X_data, self.idx, axis = 0)
        Y_data = tf.gather(Y_data, self.idx, axis = 0)

        err = Y_data - self.mean_function(X_data)

        kmm = self.kernel(X_data)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X_data, Xnew)
        kmm_plus_s = self._add_noise_cov(kmm)

        conditional = gpflow.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm_plus_s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var
    
def concat(*args):
    r = args[0]
    for a in args[1:]:
        r =  r +  a
    return r
def sumArgs(*args):
    r = args[0]
    for a in args[1:]:
        r = r + a
    return r


class GPRKernel(gpflow.models.GPR):
    """
    Use the kernel trick on Y.
    
    Only works if Ky is SPD ie, if the dimension is big
    """
    def __init__(self,
        data: gpflow.models.training_mixins.RegressionData,
        idx : tf.Tensor,
        kernel: gpflow.kernels.Kernel,
        kernelY: gpflow.kernels.Kernel = None,         
        mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
        noise_variance: float = 1.0,
        weight: float = 1.
    ):
        self.idx = idx
        self.weight = weight
        if kernelY is None:
            self.kernely = gpflow.kernels.Linear()
        else:
            self.kernely = kernelY
        super().__init__(data, kernel, mean_function, noise_variance)
        X, Y = self.data
        X = tf.gather(X, self.idx, axis = 0)
        Y = tf.gather(Y, self.idx, axis = 0)

        self.X_train, self.Y_train = X, Y
        m = self.mean_function(X)

        self.Ky = self.kernely(Y - m)
        self.Ly = tf.linalg.cholesky(self.Ky)
    
    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        # TODO : use a better way of separate gather from the GP        
        Kx = self.kernel(self.X_train)
        num_data = tf.shape(self.X_train)[0]
        k_diag = tf.linalg.diag_part(Kx)
        s_diag = tf.fill([num_data], self.likelihood.variance)
        ks = tf.linalg.set_diag(Kx, k_diag + s_diag)        
        
        # Compute 
        # Tr(Kx^{-1} Ky) using the cholesky decomposition
        # Tr((Lx Lx^t)^{-1} (Ly Ly^t)) = Tr( Ly^t (Lx Lx^t)^{-1} Ly )   [USING CYCLIC PROPERTY OF TRACE]
        # Tr( Ly^t (Lx Lx^t)^{-1} Ly ) = Tr( Ly^t Lx^{-t} Lx^{-1} Ly )    [(AB)^{-1} = B^{-1} A^{-1}}]
        # Tr((Lx^{-1}) Ly)^t (Lx^{-1} Ly) ) = Frobenius(Lx^{-1} Ly)    [Definition of trace and Frobenius norm]
        
        Lx = tf.linalg.cholesky(ks)
        L =  tf.linalg.triangular_solve(Lx, self.Ly, lower=True)   
            
        log_prob = - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(Lx)))* self.Y_train.shape[1]/2 # Logarithm
        log_prob -= tf.reduce_sum(tf.square(L))/2  
        log_prob -= tf.constant( np.log(2 * np.pi)) * (self.Y_train.shape[1] * self.Y_train.shape[0])/2 # Constant
        

        return tf.math.scalar_mul(self.weight,  log_prob)
    
class GPRFixedData(GPRMissingData):
    """
    Wrapper to make it possible to work with nans
    """
    def __init__(self,
        fixedData : tf.Variable,
        data: gpflow.models.training_mixins.RegressionData,
        idx : tf.Tensor,
        kernel: gpflow.kernels.Kernel,
        mean_function: Optional[gpflow.mean_functions.MeanFunction] = None,
        noise_variance: float = 1.0,
        weight: float = 1.,
    ):
        self.fixedData = fixedData
        self.scales = gp.Parameter(np.ones(fixedData.shape[1]), transform=gp.utilities.positive()) 
        #set_trainable(self.scales, False) # Set to false, since it produces a lot of  keeping 

        super().__init__(data, idx, kernel, mean_function, noise_variance, weight)
    
    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        
        Xfixed = tf.gather(self.fixedData * self.scales, self.idx, axis = 0)
        X, Y = self.data
        X = tf.gather(X, self.idx, axis = 0)
        Y = tf.gather(Y, self.idx, axis = 0)

        Xconcat = tf.concat([Xfixed, X], axis = 1)
        return super().log_marginal_likelihood(Xconcat, Y)
