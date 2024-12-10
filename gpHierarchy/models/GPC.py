import gpflow, tensorflow as tf
import numpy as np, sklearn
import logging
import gpflow.ci_utils
from sklearn.utils.multiclass import unique_labels

class MiniBatchWeight:
    def __init__(self, X, y, weights, batch_size ):
        self.X =X
        self.y =y
        self.weights = weights / np.sum(weights)
        self.batch_size = batch_size
        self.N = len(X)
        
    def generate(self):
        """
        Tensorflow API needs to be yield element by element. Does an internal batch to avoid repetition. It will give repeated elements if the batch size is different.
        """
        while True:
            #Sample batch
            idx = np.atleast_1d(np.random.choice(self.N, self.batch_size, replace = False, p = self.weights))
            for i in idx:
                yield self.X[i], np.atleast_1d(self.y[i])
                
                
    
class GPC(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    def __init__(self, restartOnEachFit = False, 
                 ard = True, minibatch_size = 20, maxiter = 2000, num_inducing_variables = 50, optimize_inducing = False):
        self.fullRestartOnEachFit = restartOnEachFit
        self.preparedForOptimisation = False
        self.ard = ard
        
        #Optimisation parameters
        self.minibatch_size = minibatch_size
        self.maxiter = maxiter
        self.num_inducing_variables = num_inducing_variables
        self.optimize_inducing = optimize_inducing
        
    def fit(self, X, y, w = None, debug = False, maxiter = None, minibatch_size = None):
        if debug:
            logging.info('Starting new optimisation')

        if not self.preparedForOptimisation or self.fullRestartOnEachFit:
            self.prepareModel(X)
        if maxiter is  None:
            maxiter = self.maxiter
        if minibatch_size is None:
            minibatch_size = self.minibatch_size

            
        self.classes_ = unique_labels(y)
        w = w if w is not None else np.ones(X.shape[0])
        
        generator = MiniBatchWeight(X, y, w , minibatch_size )
        training_data = tf.data.Dataset.from_generator(generator.generate, (tf.float64, tf.float64))

        maxiter = gpflow.ci_utils.ci_niter(maxiter)
        self.logf = self.run_adam(maxiter, training_data, minibatch_size)
        
    def run_adam(self, iterations, training_data, minibatch_size):
        """
        Utility function running the Adam optimizer

        :param model: GPflow model
        :param interations: number of iterations
        """
        model = self.m
        # Create an Adam Optimizer action
        logf = []
        train_iter = iter(training_data.batch(minibatch_size))
        training_loss = model.training_loss_closure(train_iter, compile=True)
        optimizer = tf.optimizers.Adam()

        @tf.function
        def optimization_step():
            optimizer.minimize(training_loss, model.trainable_variables)
        for step in range(iterations):
            optimization_step()
            if step % 10 == 0:
                elbo = -training_loss().numpy()
                logf.append(elbo)
        return logf
    
    def prepareModel(self, X, num_inducing_variables = None, optimize_inducing = None):
        if num_inducing_variables is None:
            num_inducing_variables = self.num_inducing_variables
        if optimize_inducing is None:
            optimize_inducing = self.optimize_inducing
            
        M = num_inducing_variables
        N = len(X)
        kernel = gpflow.kernels.SquaredExponential(lengthscales= np.std(X, axis = 0)) + gpflow.kernels.White(1e-2)
        inducing_idx = np.random.choice(N, M, replace = False)
        Z = X[inducing_idx, :].copy()  # Initialize inducing locations to the first M inputs in the dataset

        m = gpflow.models.SVGP(kernel, gpflow.likelihoods.Bernoulli(), Z, num_data= N)
        self.m = m
        # We turn off training for inducing point locations
        gpflow.set_trainable(self.m.inducing_variable, optimize_inducing)
        self.preparedForOptimisation = True
        
    def predict_proba(self, X):
        t = self.m.predict_y(X)[0]
        return np.column_stack((1 - t, t))
    
    def predict(self, X):
        return self.predict_proba(X)[:, 1] > 0.5