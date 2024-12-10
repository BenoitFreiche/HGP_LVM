import tensorflow as tf, tensorflow_probability as tfp
import gpflow
import numpy as np

class SampleWeightModel:
    """
    Optimise the sample weights, in a similar fashion to boosting
    """
    def __init__(self, X, w0, lengthscale = 0.5, penalty = 0, trainLengthscale = False):
        w0 = w0.reshape((-1, 1))
        self.lengthscale = lengthscale
        self.penalty = penalty
        
        
        self.f_variable = tf.Variable(np.log(w0/(1- w0)), dtype= tf.float64)
        self.X_variable = tf.Variable(X, dtype = tf.float64,trainable = False)

        kernelRBF = gpflow.kernels.RBF() 
        kernelBias = gpflow.kernels.Bias()
        self.gp = gpflow.models.GPR((self.X_variable, self.f_variable), kernel = kernelRBF )
        kernelBias.variance.prior = tfp.distributions.Normal(gpflow.utilities.to_default_float(0), gpflow.utilities.to_default_float(1))
        kernelRBF.lengthscales.assign(self.lengthscale)
        gpflow.utilities.set_trainable(kernelRBF.lengthscales, trainLengthscale)
        gpflow.utilities.set_trainable(kernelRBF.variance, False)
        gpflow.utilities.set_trainable(kernelBias.variance, True)
        self.gp.likelihood.variance.assign(1e-2)
        
    @tf.function
    def loss(self):
        gp_prior_loss =  -self.gp.log_posterior_density()
        W = tf.math.sigmoid(self.f_variable)
        loss_W = - tf.linalg.tensordot(self.E, W, 1)
        loss_prior_W =  - tf.reduce_sum(tf.math.log(W)) # Penalise predictions near 0
        loss_prior_W =  tf.reduce_sum(1 - W) # Penalise predictions near 0

        return  loss_W  + gp_prior_loss + self.penalty * loss_prior_W

    def fit(self, X, E, w0 = None, n_iter = 1000):
        ##
        # maximise W * E + log P(f), where P(f) is the gaussian prior
        #
        # Derivation from P(w | X) = P(X | w) P(w)/P(X) ; but after doing the logarithms, the factor P(X) is not important since it does not depend on W
        ##

        if w0 is not None:
            self.f_variable.assign(np.log(w0/(1- w0)))
        self.X_variable.assign(X)
        self.E = tf.constant(E, dtype = tf.float64)
    
        opt = gpflow.optimizers.Scipy()
        maxiter = gpflow.ci_utils.ci_niter(n_iter)

        # Store the results for debugging
        self.resultTrain = opt.minimize(
            self.loss, 
            variables=  self.gp.trainable_variables,
            compile = True,
            options=dict(maxiter=maxiter),
            method = 'L-BFGS-B'
        )
        return self
    
    def predict_proba(self, X):
        fpred , _ = self.gp.predict_f(X,)
        y1 = tf.math.sigmoid(fpred).numpy().flatten()
        return np.column_stack([1 -y1, y1])