"""
B. Freiche 04-2024
Code for hierarchical GP-LVM applied to celebA
Freiche et al. "Hierarchical data integration with Gaussian processes: application to the characterization of cardiac ischemia-reperfusion pattern" 2024
"""

# IMPORTS
import time
from gpHierarchy.models import mskr
from gpHierarchy.base import stablePCA, training
from gpHierarchy.models import easyGP, gpLVM
import pyvista, importlib,sklearn.metrics, pandas, numpy as np, sklearn, collections
from gpflow.models import BayesianGPLVM
from sklearn.decomposition import PCA
from gpHierarchy.base import hierarchyGraph
import sys
import pickle
import numpy as np
from scipy.spatial.distance import pdist as pairwise_distances
from types import SimpleNamespace
from numpy.linalg import inv
import os
import copy
import tensorflow as tf
from gpHierarchy.base import hierarchyGraph, initialisation, training
from gpflow.utilities import set_trainable
import argparse
from scipy.linalg import eigh
import pandas as pd
import matplotlib.pyplot as plt

#%%
os.chdir('C:/Users/b_freich/Documents/Research/Code/Code_these_pour_ND/data/celebA_brunes')


attr = pd.read_csv('list_attr.csv')
bbox = pd.read_csv('list_bbox.csv')
partition = pd.read_csv('list_eval_partition.csv')
landmarks = pd.read_csv('list_landmarks.csv')

#%% Recupération données
images = []
key_im = []
n_image = 4714
egal = True
if egal:
    dir_ = 'images_egalisees'
else :
    dir_ = 'images'
#outliers = [40,119,198,239,394,416]
seed = 50 #30
np.random.seed(seed)
list_index = np.arange(n_image)
list_names = np.asarray(np.sort(os.listdir(dir_)))[list_index]
#s = []
for image in list_names:
    im_ = plt.imread(os.path.join(dir_,image))[:,:,0]
    images.append(im_.astype(int))   
    key_im.append(image)

boolean = [landmark_key in key_im for landmark_key in landmarks.values[:,0]]

land = landmarks.loc[boolean]
land = land.values[:,1:].astype(float)

at = attr.loc[boolean]
at = at.values[:,1:]

N = 400
data_images = np.asarray((images)[:N])/255
data_reshape = data_images.reshape((400,-1))

land = land[:N]
land = land/np.max(land)
    
#%%
path_save = 'C:/Users/b_freich/Documents/Research/Code/Code_these_pour_ND/Results/LANDMARKS/dim_3'
dim = 3

importlib.reload(gpLVM)
importlib.reload(gpLVM.initialisation)
importlib.reload(gpLVM.hierarchyGraph)
importlib.reload(gpLVM.training)

nodeX0_copy = np.nan

Y_train = copy.copy(land)

n_samples = len(Y_train)
print('Training for ', n_samples,' samples \n')

omega_2 = 1
n_0 = dim
ls = [1]*dim
kervar = 1e-2
likevar = 1e-6+1e-9
it = 200
    
to_save = dict()

model_landmarks = hierarchyGraph.Graph()
trainLengthscalesBetweenLevels = False
with model_landmarks:            
    nodeX = hierarchyGraph.LatentVariable(size = (n_samples,n_0),name = 'X',
                                        prior = hierarchyGraph.getGaussianPrior(n_0, omega_2))
    nodeX.trainable = True
    
    nodeY = hierarchyGraph.ObservedVariable(X = Y_train, varianceType= 'automatic', name = 'Y', removeMean = True)
    
    hierarchyGraph.Link(nodeX, nodeY,kernel = 'rbf',weight = 1,lengthscales = ls)    

model_landmarks.setGP( nodeX, nodeY, likelihood =  likevar,variance = kervar)

model_landmarks.niter = 0
gp = model_landmarks.getGP(nodeX, nodeY) 

set_trainable(gp.kernel.variance,True) 
set_trainable(gp.kernel.lengthscales,True)
set_trainable(gp.likelihood.variance,True)    

print('initialisation model...')
initialisation.initialiseLatentVariablesPCA(model_landmarks)

# Pour sauvegarder uniquement le dernier modele:
argsOpt_1 = {'mode' : 'sequence', 'steps' : training.getTrainStrategyAlternate(it)}
model_landmarks.train(**argsOpt_1)
to_save['X'] = [nodeX.X.numpy()]
to_save['kernel_variance'] = [gp.kernel.variance] 
to_save['kernel_lengthscales'] = [gp.kernel.lengthscales]
to_save['likelihood_variance'] = [gp.likelihood.variance]


# Si on veut sauvegarder le modele a chaque iterations:
    
# to_save['X'] = []
# to_save['kernel_variance'] = [] 
# to_save['kernel_lengthscales'] = []
# to_save['likelihood_variance'] = []
# for _ in range(it):
#     argsOpt_1 = {'mode' : 'sequence', 'steps' : training.getTrainStrategyAlternate(1)}
#     model_landmarks.train(**argsOpt_1)
#     to_save['X'].append(nodeX.X.numpy())
#     to_save['kernel_variance'].append(gp.kernel.variance) 
#     to_save['kernel_lengthscales'].append(gp.kernel.lengthscales)
#     to_save['likelihood_variance'].append(gp.likelihood.variance)

with  open(os.path.join(path_save,'spaces.pkl'),'wb') as f:
    pickle.dump(to_save,f)
    
del nodeX,nodeY,model_landmarks,to_save,gp
#%%
path_save = 'C:/Users/b_freich/Documents/Research/Code/Code_these_pour_ND/Results/IMAGES/dim_10/'
dim = 10

nodeX0_copy = np.nan


Y_train = copy.copy(data_reshape)

    
importlib.reload(gpLVM)
importlib.reload(gpLVM.initialisation)
importlib.reload(gpLVM.hierarchyGraph)
importlib.reload(gpLVM.training)

n_samples = len(Y_train)
print('Training for ', n_samples,' samples \n')

omega_2 = 1
n_0 = dim
ls = [1]*dim
kervar = 1e-1
likevar = 1e-2
it = 200
    
to_save = dict()

model_images = hierarchyGraph.Graph()
trainLengthscalesBetweenLevels = False
with model_images:            
    nodeX = hierarchyGraph.LatentVariable(size = (n_samples,n_0),name = 'X',
                                        prior = hierarchyGraph.getGaussianPrior(n_0, omega_2))
    nodeX.trainable = True
    
    nodeY = hierarchyGraph.ObservedVariable(X = Y_train, varianceType= 'automatic', name = 'Y', removeMean = True)
    
    hierarchyGraph.Link(nodeX, nodeY,kernel = 'rbf',weight = 1,lengthscales = ls)    

model_images.setGP( nodeX, nodeY, likelihood =  likevar,variance = kervar)

model_images.niter = 0
gp = model_images.getGP(nodeX, nodeY) 

set_trainable(gp.kernel.variance,True) 
set_trainable(gp.kernel.lengthscales,True)
set_trainable(gp.likelihood.variance,True)    

print('initialisation model...')
initialisation.initialiseLatentVariablesPCA(model_images)
argsOpt_1 = {'mode' : 'sequence', 'steps' : training.getTrainStrategyAlternate(it)}
model_images.train(**argsOpt_1)
to_save['X'] = [nodeX.X.numpy()]
to_save['kernel_variance'] = [gp.kernel.variance] 
to_save['kernel_lengthscales'] = [gp.kernel.lengthscales]
to_save['likelihood_variance'] = [gp.likelihood.variance]

with  open(os.path.join(path_save,'spaces.pkl'),'wb') as f:
    pickle.dump(to_save,f)
 
del nodeX,nodeY,model_images,to_save,gp

#%% TRAINING HIERARCHY = need to have trained LANDMARKS and IMAGES first
path_save = 'C:/Users/b_freich/Documents/Research/Code/Code_these_pour_ND/Results/HIERARCHY/dim_10/link_1/'
link = 1
dim = 10


Y_train_images = copy.copy(data_reshape)
Y_train_land = copy.copy(land)

importlib.reload(gpLVM)
importlib.reload(gpLVM.initialisation)
importlib.reload(gpLVM.hierarchyGraph)
importlib.reload(gpLVM.training)

n_samples = len(Y_train_images)
print('Training for ', n_samples,' samples \n')

with open('C:/Users/b_freich/Documents/Research/Code/Code_these_pour_ND/Results/IMAGES/dim_10/spaces.pkl','rb') as f:
    out_images = pickle.load(f)

kernel_lengthscales_images = [np.mean(out_images['kernel_lengthscales'][0].numpy())]*dim
# kernel_lengthscales_mvo = [3]*dim
kernel_variance_images = out_images['kernel_variance'][0].numpy()
likelihood_variance_images = 1e-2

with open('C:/Users/b_freich/Documents/Research/Code/Code_these_pour_ND/Results/LANDMARKS/dim_3/spaces.pkl','rb') as f:
    out_land = pickle.load(f)
    
space_land = out_land['X'][0]

space_land = np.concatenate([space_land,np.zeros((400,7))],axis = 1)

    
dim_land =10

kernel_lengthscales_land = [np.mean(out_land['kernel_lengthscales'][0].numpy())]*dim_land
kernel_variance_land = out_land['kernel_variance'][0].numpy()
likelihood_variance_land = out_land['likelihood_variance'][0].numpy() 

to_save = dict()
model = hierarchyGraph.Graph()
trainLengthscalesBetweenLevels = False
with model:
    nodeY1 = hierarchyGraph.ObservedVariable(X = Y_train_images, varianceType= 'automatic', name = 'Y1', removeMean = True)
    
    nodeX0 = hierarchyGraph.LatentVariable(size = (n_samples,dim_land),name = 'X0',
                                        prior = hierarchyGraph.getGaussianPrior(dim, 1))
    nodeX0.trainable = True

    nodeX1 = hierarchyGraph.LatentVariable(size = (n_samples,dim), name = 'X1',
                                        prior = hierarchyGraph.getGaussianPrior(dim, 1))
    
    nodeY0 = hierarchyGraph.ObservedVariable(X = Y_train_land, varianceType= 'automatic', name = 'Y0', removeMean = True)
    
    hierarchyGraph.Link(nodeX0, nodeY0,kernel = 'rbf',weight = 1,lengthscales = kernel_lengthscales_land)    
    hierarchyGraph.Link(nodeX1, nodeY1,kernel = 'rbf',weight = 1,lengthscales = kernel_lengthscales_images) 

model.setGP( nodeX0, nodeY0, likelihood =  likelihood_variance_land,variance = kernel_variance_land)
model.setGP( nodeX1, nodeY1, likelihood =  likelihood_variance_images,variance = kernel_variance_images)

model.niter = 0
gp_1 = model.getGP(nodeX0, nodeY0) 
gp_2 = model.getGP(nodeX1, nodeY1)

print('initialisation model...')
initialisation.initialiseLatentVariablesPCA(model)

print('X0 trained')
nodeX0.setValues(space_land) 
   
print('\n \n Training only X1 ... ')
set_trainable(gp_2,True)   
set_trainable(gp_1,False)   

set_trainable(gp_2.kernel.variance,False) 
set_trainable(gp_2.kernel.lengthscales,False)
set_trainable(gp_2.likelihood.variance,False)

with model:
    nodeX1.trainable = True
    nodeX0.trainable = False
    #ON a modifié ici, on met les paramètres de INFARCT SUR le lien X0 X1
    hierarchyGraph.Link(nodeX0, nodeX1,kernel = 'rbf',weight = link,lengthscales = kernel_lengthscales_land) 
    model.setGP( nodeX0, nodeX1, likelihood =  likelihood_variance_land,variance = kernel_variance_land)
    gp_3 = model.getGP(nodeX0, nodeX1)
    
    hierarchyGraph.Link(nodeX1, nodeX0,kernel = 'rbf',weight = link,lengthscales = kernel_lengthscales_images) 
    model.setGP( nodeX1, nodeX0, likelihood =  likelihood_variance_images,variance = kernel_variance_images)
    gp_4 = model.getGP(nodeX1, nodeX0)

    set_trainable(gp_3.kernel.variance,False) 
    set_trainable(gp_3.kernel.lengthscales,False)
    set_trainable(gp_3.likelihood.variance,False)

    set_trainable(gp_4.kernel.variance,False) 
    set_trainable(gp_4.kernel.lengthscales,False)
    set_trainable(gp_4.likelihood.variance,False)    
    

print('initialisation model...')
initialisation.initialiseLatentVariablesPCA(model)
print('X0 trained')
nodeX0.setValues(space_land) 
# argsOpt_1 = {'mode' : 'sequence', 'steps' : training.train()}
print('training')
model.train()
to_save['X'] = [nodeX1.X.numpy()]

with  open(os.path.join(path_save,'spaces.pkl'),'wb') as f:
    pickle.dump(to_save,f)
