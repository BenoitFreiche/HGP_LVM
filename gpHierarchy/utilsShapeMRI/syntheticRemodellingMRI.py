import sklearn.decomposition, numpy as np
import scipy,sys, pyvista, vtk, collections
from vtk.util import numpy_support
from . import  meshUtils, generateMeasurementsMRI
from ..models import easyGP, mskr

class SyntheticPopulationMRI:
    def __init__(self, Y, n_components_pca = .95, sampleMesh= None, samplingMode = 'pca', labels = None):
        self.pca = sklearn.decomposition.PCA(n_components=n_components_pca)
        self.pca.fit(Y)
        self.Y = Y
        self.n_pca = len(self.pca.components_)
        self.transformations = []
        self.sampleMesh = sampleMesh
        self.labels = labels
        self.samplingMode = samplingMode
        
    def generateSamples(self, n = 1 ):
        if self.samplingMode == 'pca':
            Y = self.pca.inverse_transform(np.random.normal(scale = np.sqrt(self.pca.explained_variance_), size = (n,self.n_pca)))
        elif self.samplingMode == 'sample':
            idx = np.random.choice(self.Y.shape[0], n, replace = False)
            Y = self.Y[idx].copy() 
        else:
            raise ValueError('sampleMode not recognised')
            
        params = {}
        for f in self.transformations:
            Y = f(Y)
        return Y
    
    def registerTransformation(self, f):
        self.transformations.append(f)
        
def getLA(Y_3D, labels):
    apexPos = np.mean(Y_3D[labels['ApexLVEndo']], axis = 0)
    mitralMean = np.mean(Y_3D[labels['MitralValve']], axis = 0)
    la = mitralMean - apexPos
    lav = la /np.linalg.norm(la)
    return lav, np.linalg.norm(la)

def addWhiteNoise(Y, scale = 1):
    return Y + np.random.normal(size = Y.shape, scale = scale)

    
class NoiseModelGP:
    def __init__(self, sampleMesh, scale = 1, omega = 10, nEigs = 50):
        self.scale = scale
        self.omega = omega
        self.nEigs = nEigs
        Y_3D = sampleMesh.points
        d = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(Y_3D, metric = 'euclidean'))**2
        K = self.scale *np.exp(-d/(self.omega**2)) + self.scale/25 * np.eye(len(d))
        if False:
            K[K < 1e-2] = 0
            K = scipy.sparse.bsr_matrix(K)
            self.v, self.E =scipy.sparse.linalg.eigsh(K, k =self.nEigs)
        else:
            K = self.scale *np.exp(-d/(self.omega**2)) + self.scale/25 * np.eye(len(d))
            v, E = np.linalg.eigh(K)
            self.v, self.E = v[-self.nEigs:], E[:, -self.nEigs :]
        
    def addNoise(self, Y):
        noise = self.E @ (np.sqrt(self.v) *np.random.normal(size = (3 * len(Y), self.nEigs))).T
        noise = noise.T.reshape( (len(Y), -1 ), order = 'F')
        return Y + noise
    
    def applyTransform(self, Y):
        return self.addNoise(Y), None
class NoiseModelWhite:
    def __init__(self, scale = 1):
        self.scale = scale
        
    def addNoise(self, Y):
        noise = self.scale * np.random.normal(scale = self.scale, size = Y.shape)
        return Y + noise
    
    def applyTransform(self, Y):
        return self.addNoise(Y), None

numpyToVTK = lambda x, sampleMesh: pyvista.PolyData(x.reshape((-1, 3)), sampleMesh.faces)


from abc import ABC, abstractmethod
def getLAApprox(mesh):
    v = mesh.points[4310] - mesh.points[953]
    length = np.linalg.norm(v)
    v /= length
    return v, length

class RemodellingBase(ABC):            
    def remodelPopulation(self, Y, t = None):
        if t is None:
            t = np.random.rand(len(Y))
        return np.array([self.remodelling(y, ti) for y, ti in zip(Y, t)]), t
    @abstractmethod
    def remodelling(self, Y, t):
        pass
    
    def getRemodellingDirection(self, y, eps = 1e-6):
        y_remod  = self.remodelling(y, eps)
        # Allign y_remod to y
        y_remod = meshUtils.procrustes_single_mesh(y_remod, y)
        v = y_remod -y
        v /= np.linalg.norm(v)
        return v
    def applyTransform(self, Y):
        return self.remodelPopulation(Y)

def computeNormalsCorrectOrientationLV(mesh, labels, returnInversions = False):
    mesh = mesh.compute_normals(split_vertices = False, point_normals= False, consistent_normals = False)
    triangles = mesh.faces.reshape((-1, 4))[:, 1:]
    triangleCenters = np.mean(mesh.points[triangles], axis = 1)

    mitral = meshUtils.extract_faces(mesh, labels['MitralValve'])
    mitralCenter = np.mean(numpy_support.vtk_to_numpy(mitral.GetPoints().GetData()), axis = 0)

    normals =numpy_support.vtk_to_numpy(mesh.GetCellData().GetArray('Normals'))
    triangleCenters = np.mean(mesh.points[triangles], axis = 1)
    inversions = np.zeros(len(normals))
    for i, n in enumerate(normals):
        if i in labels['LeftVentricle']:
            sign = np.sign(np.dot(n, mitralCenter - triangleCenters[i]))
        elif i in labels['LVMyo'] and i not in labels['LeftVentricle']:
            sign = -np.sign(np.dot(n, mitralCenter - triangleCenters[i]))
        else:
            sign = 1
            
        if sign <= 0:
            normals[i] *= sign
        inversions[i] = sign
        
    if returnInversions:
        return inversions
    mesh.GetCellData().RemoveArray('Normals')
    normalsVTK = numpy_support.numpy_to_vtk(normals)
    normalsVTK.SetName('Normals')
    mesh.GetCellData().AddArray(normalsVTK)

    f = vtk.vtkCellDataToPointData()
    f.SetInputData(mesh)
    f.Update()
    mesh = pyvista.PolyData(f.GetOutput())
    mesh.GetCellData().AddArray(normalsVTK)
    return mesh
    
def cleanOrientationMesh(sampleMesh, labels):
    signs = computeNormalsCorrectOrientationLV(sampleMesh, labels, True)
    triangles = sampleMesh.faces.reshape((-1, 4))[:, 1:]
    for i in np.where(signs == -1)[0]:
        triangles[i, 0], triangles[i, 1] = triangles[i, 1], triangles[i, 0]
    faces = np.concatenate([np.ones((len(triangles) , 1), dtype = int) * 3, triangles], axis = 1).flatten()
    return pyvista.PolyData(sampleMesh.points , faces)

class RemodellingPoint(RemodellingBase):
    def __init__(self, pointId, sampleMesh, labels, omega_ref = 15, scale = 1, normalComputation = 'pyvista'):
        self.sampleMesh = sampleMesh
        self.omega_ref = omega_ref
        _, self.la_ref = getLAApprox(sampleMesh)
        self.scale = scale
        self.pointId = pointId
        self.labels = labels
        self.normalComputation = normalComputation
        
    def remodelling(self, Y_3D, t):            
        # Compute normals
        mesh = numpyToVTK(Y_3D, self.sampleMesh)
        if self.normalComputation == 'pyvista':
            mesh = mesh.compute_normals( consistent_normals = False)
        elif self.normalComputation == 'verifyOrientation':
            mesh = computeNormalsCorrectOrientationLV(mesh, self.labels)
        else:
            raise ValueError()
        # Compute omega
        _, la =  getLAApprox(mesh)
        omega = self.omega_ref *  la / self.la_ref

        ds = scipy.spatial.distance_matrix( mesh.points[self.pointId].reshape((-1, 3)) , mesh.points)**2
        
        k = self.scale * np.exp(-ds/omega**2).reshape((-1, 1)) 
        #utils.meshUtils.addArrayVTK(mesh, k  * mesh.point_normals, name = 'Kvector')
        displacement = t * k * numpy_support.vtk_to_numpy(mesh.GetPointData().GetArray('Normals'))
        return (mesh.points + displacement).flatten()
    

    # For pickle
    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass
    
class RemodellingScaling(RemodellingBase):
    def __init__(self, scale):
        self.scaleLogarithmic = 2 * np.log(1 + scale)
    def remodelling(self, Y_3d, t):
        return Y_3d * np.exp((t - 0.5) * scaleLogarithmic)
    
class RemodellingFreeWallSingle(RemodellingPoint):
    def __init__(self, sampleMesh, labels, **kwargs):
        super().__init__(1106,  sampleMesh,labels, **kwargs)


class RemodellingSphericalDilationSingle(RemodellingBase):
    def __init__(self, labels = None , scale = 1):
        self.scale = scale
        if labels is not None:
            self.labels = labels
            self.getLA = lambda Y_3D: getLA(Y_3D, self.labels)
        else:
            self.getLA = lambda Y_3D: getLAAprox(Y_3D)
    def remodelling(self, Y,t):
    
        # Compute LA
        Y_3D = Y.reshape((-1, 3))
        la, _ = self.getLA(Y_3D)

        dilatation = np.eye(3) + self.scale * t * (np.eye(3) - np.outer(la, la))
        return (Y_3D @ dilatation).flatten()
    
class RemodellingSphericalDilationConstantVolumeSingle(RemodellingBase):
    def __init__(self, labels , scale = 1):
        self.scale = scale
        self.labels = labels

    def remodelling(self, Y, t):
        """
        Add sphericity, without modifying the volumes: that should be discarded by the hierarchy 
        """
        # Compute LA
        Y_3D = Y.reshape((-1, 3))
        la, _ = getLA(Y_3D, self.labels)

        dilatation = (np.eye(3) + self.scale * t * (np.eye(3) - np.outer(la, la)))
        shrinking =  (np.eye(3) - np.outer(la, la)) + 1/(1 + self.scale * t)**2 * np.outer(la, la)
        return (Y_3D @ dilatation @ shrinking ).flatten()
    
    
    
def generateDataset(synthGenerator,remod, noiseModel, labels, n_samples, seed = None):
    if seed is not None:
        np.random.seed(seed)

    Y_synthetic = synthGenerator.generateSamples(n_samples)
    Y_synthetic_remodelled, t = remod.remodelPopulation(Y_synthetic)
    Y_synthetic_remodelled_procrustes, _ = meshUtils.procrustes(Y_synthetic_remodelled)

    Y_synthetic_remodelled_noisy = noiseModel.addNoise(Y_synthetic_remodelled_procrustes)
    Y_synthetic_remodelled_noisy, _ = meshUtils.procrustes(Y_synthetic_remodelled_noisy)
    measurementsFull = generateMeasurementsMRI.generateMeasurements(Y_synthetic_remodelled_noisy,
                                                             meshSample=synthGenerator.sampleMesh, labelsBiventricular= labels)

    return Y_synthetic_remodelled_noisy, t, measurementsFull


class Generator:
    def __init__(self, generator, transformations):
        self.generator_0 = generator
        self.transformations = transformations
    def generate(self, n_samples):
        self.remodellingRandomVariables = {}
        Y = self.generator_0.generateSamples(n_samples)
        for transformer in self.transformations:
            Y, t = transformer.applyTransform(Y)
            if t is not None:
                self.storeRandomVariableSample(t, transformer.name)
        Y, _ = meshUtils.procrustes(Y)

        return Y, self.remodellingRandomVariables
    def storeRandomVariableSample(self, t, name):
        self.remodellingRandomVariables[name] = t
        
resultDotProduct = collections.namedtuple('ResultDotProduct', 'dotProduct vGroundtruth vReconstructed')
def computeDotProductStreamlineGroundtruth(x, Y, t, remod, eps = 1e-3, gpReconstruction = None, gp = None):
    """
    Computes the dot product between the streamlines of the predicted remodelling (fitted via a GP), and the theoretical remodelling obtained by applying the remodelling generator to each mesh. 
    """
    if gp is None:
        gp = easyGP.EasyGP(x.shape[1], mode = 'streamline')
        gp.fit(x, t)
    if gpReconstruction is None:
        gpReconstruction  = mskr.MKSRWrapper()
        gpReconstruction.fit(x, Y )
    
    r = np.zeros(len(Y))
    vReconstructed = np.zeros_like(Y)
    vTheo = np.zeros_like(Y)
    grads_t,_ = gp.m.predict_jacobian(x)
    x_plus = x + eps * grads_t.reshape(x.shape)
    y0  = gpReconstruction.predict(x)
    y1  = gpReconstruction.predict(x_plus)

    #Can be vectorised for more efficiency
    for i, xi in enumerate(x):
        #if gradient is 0, set to nan
        vReconstructed[i] = (y1[i] - y0[i])/np.linalg.norm(y1[i] - y0[i])
        vTheo[i] = remod.getRemodellingDirection(y0[i])
        r[i] = np.dot(vTheo[i], vReconstructed[i])
    return resultDotProduct(r, vTheo, vReconstructed)
