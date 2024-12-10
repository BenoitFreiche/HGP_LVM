import collections, networkx , numpy as np, sklearn.impute, sklearn.manifold
from gpflow.utilities import ops, print_summary
from .hierarchyGraph import LatentVariable, Graph
from .stablePCA import StablePCA, stableEmbeddings
def resetLatentVariables(model):
    for n in model.nodes.values():
        if isinstance(n, LatentVariable):
            n.initialised = False

def initialiseNodePCA(model, nodeName, **kwargs):
    node = model.nodes[nodeName]
    if node.initialised:
        return

    Y = np.zeros([node.X.shape[0],0])
    for a in kwargs.get('adjacency', model.adjacency)[nodeName]:
        if model.countInitialisation__ >= len(model.gpLinkFunctions):
            raise ValueError('There seems to be a cycle with latent variables')
        if isinstance(model.nodes[a], LatentVariable) and model.nodes[a].initialised == False:
            model.countInitialisation__ += 1
            initialiseNodePCA(model, a, **kwargs)
        if not isinstance(model.nodes[a], LatentVariable) or not kwargs.get('use_Y_only', False):
            Y = np.concatenate((Y,model.nodes[a].value), axis = 1)
    #imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    #Y = imp.fit_transform(Y)
    drClass = kwargs.get('dimensionalityReduction', StablePCA)
    dr = drClass(n_components = node.X.shape[1])
    Y_data_mean =  stableEmbeddings(dr.fit_transform(Y))
    Y_data_mean /= np.nanstd(Y_data_mean, axis = 0)

    node.setValues(Y_data_mean)
    node.initialised = True
    

def initialiseLatentVariablesPCA(model, leafToRoot = True, **kwargs):
    """
    Initialise the weights using PCA and the graph structure
    """
    model.initialised = True
    model.resetLatentVariables()
    model.countInitialisation__ = 0
    for n in model.latentVariables.keys():
        initialiseNodePCA(model, n, adjacency = model.adjacency if leafToRoot else model.inverseAdjacency, **kwargs)

def initialiseNodeDM(model, nodeName, **kwargs):
    node = model.nodes[nodeName]
    if node.initialised:
        return

    Y = np.zeros([node.X.shape[0],0])
    for a in kwargs.get('adjacency', model.adjacency)[nodeName]:
        if model.countInitialisation__ >= len(model.gpLinkFunctions):
            raise ValueError('There seems to be a cycle with latent variables')
        if isinstance(model.nodes[a], LatentVariable) and model.nodes[a].initialised == False:
            model.countInitialisation__ += 1
            initialiseNodePCA(model, a, **kwargs)
            
        if not isinstance(model.nodes[a], LatentVariable) or not kwargs.get('use_Y_only', False):
            Y = np.concatenate((Y,model.nodes[a].value), axis = 1)
    #imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    #Y = imp.fit_transform(Y)
    drClass = kwargs.get('dimensionalityReduction', StablePCA)
    dr = drClass(n_components = node.X.shape[1])
    Y_data_mean =  stableEmbeddings(dr.fit_transform(Y))
    Y_data_mean /= np.nanstd(Y_data_mean, axis = 0)
    
    node.setValues(Y_data_mean)
    node.initialised = True

def initialiseLatentVariablesDM(model, leafToRoot = True, **kwargs):
    """
    Initialise the weights using PCA and the graph structure
    """
    model.initialised = True
    model.resetLatentVariables()
    model.countInitialisation__ = 0
    for n in model.latentVariables.keys():
        initialiseNodeDM(model, n, adjacency = model.adjacency if leafToRoot else model.inverseAdjacency, **kwargs)

def initialiseNodeRandom(model, nodeName, **kwargs):
    node = model.nodes[nodeName]
    if node.initialised:
        return

    Y = np.zeros([node.X.shape[0],0])
    for a in kwargs.get('adjacency', model.adjacency)[nodeName]:
        if model.countInitialisation__ >= len(model.gpLinkFunctions):
            raise ValueError('There seems to be a cycle with latent variables')
        if isinstance(model.nodes[a], LatentVariable) and model.nodes[a].initialised == False:
            model.countInitialisation__ += 1
            initialiseNodePCA(model, a, **kwargs)
            
        if not isinstance(model.nodes[a], LatentVariable) or not kwargs.get('use_Y_only', False):
            Y = np.concatenate((Y,model.nodes[a].value), axis = 1)
    #imp = sklearn.impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    #Y = imp.fit_transform(Y)
    drClass = kwargs.get('dimensionalityReduction', StablePCA)
    dr = drClass(n_components = node.X.shape[1])
    Y_data_mean =  stableEmbeddings(dr.fit_transform(Y))
    Y_data_mean /= np.nanstd(Y_data_mean, axis = 0)
    
    Y_init = np.random.randn(len(Y_data_mean),node.X.shape[1])
    node.setValues(Y_init)
    node.initialised = True

def initialiseLatentVariablesRandom(model, leafToRoot = True, **kwargs):
    """
    Initialise the weights using PCA and the graph structure
    """
    model.initialised = True
    model.resetLatentVariables()
    model.countInitialisation__ = 0
    for n in model.latentVariables.keys():
        initialiseNodeRandom(model, n, adjacency = model.adjacency if leafToRoot else model.inverseAdjacency, **kwargs)

def initialise(self, mode = 'pca'):
    if mode == 'pca':
        initialiseLatentVariablesPCA(self)
    elif mode == 'random':
        initialiseLatentVariablesRandom(self)
    elif mode == 'isomaps':
        initialiseLatentVariablesPCA(dr = sklearn.manifold.Isomap)
    else:
        raise ValueError(f'Unknown initialisation mode "{mode}"')
setattr(Graph, 'initialise', initialise)