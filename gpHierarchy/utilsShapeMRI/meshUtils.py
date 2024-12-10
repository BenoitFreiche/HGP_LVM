import numpy as np
import scipy, pyvista, vtk
import collections, re
from vtk.util import numpy_support
from pymeshfix import _meshfix

def readLabels(path):
    d = collections.defaultdict(list)
    i = 0
    startReading = False
    with open(path) as f:
        for l in f.readlines():
            if not startReading:
                if l.startswith('#'):
                    continue
                elif re.match('[0-9]+\s+[0-9]+', l):
                    startReading = True
            else:
                for p in l.split():
                    d[p].append(i)
                i += 1
    return d

def generate_subpart_labels(originalLabelPath, partList, mode = 'remove', outputPath = ''):
    """
    generates a label file with a subselection of parts (ie, removing the atria from the full model).
    Needs the path of the full labels, and the number of vertices and faces of the 
    TODO: use the dictionary instead of requiring the path and read it.
    Returns a dictionary, with the remaining labels.
    """

    output = []
    if mode not in ['remove', 'mantain']:
        print ('generate_subpart_labels : Unknown mode', mode)
        raise ValueError()
        
    inverseLabels = collections.defaultdict(set)
    nFacesWritten = 0
    nFaceInfo = None
    labelsNew = collections.defaultdict(list)
    with open(originalLabelPath) as f:
        for line in f.readlines():
            if line.startswith('#'):
                output.append(line.strip())
            elif line[0].isdigit():
                nFaceInfo = len(output)
                output.append('')
            else:
                if mode == 'remove':
                    labels = list(filter(lambda s: s not in partList, line.split()))
                elif mode == 'mantain':
                    labels = list(filter(lambda s: s in partList, line.split()))
                    
                if labels:
                    output.append(' '.join(labels))
                    for label in labels:
                        labelsNew[label].add(nFacesWritten)
                    nFacesWritten += 1

    output[nFaceInfo] = '%d %d' % (0, nFacesWritten)
    if outputPath:
        with open(outputPath, 'w') as f:
            f.write('\n'.join(output))
    return labelsNew


def cellToPointLabel(labels, triangles):
    labelsPoints = {}
    for l, v in labels.items():
        s = set()
        for t in v:
            for i in range(3):
                s.add(triangles[t][i])
        
        labelsPoints[l] = np.array(list(s))
    return labelsPoints

def addArrayVTK(mesh, v, name = '', domain = 'point'):
    vVTK = numpy_support.numpy_to_vtk(v)
    if name:
        vVTK.SetName(name)
    if domain == 'point':
        mesh.GetPointData().AddArray(vVTK)
    elif domain == 'cell':
        mesh.GetCellData().AddArray(vVTK)
    else:
        raise ValueError('Domain not understood')    

def addEpiToLabels(labels, sampleMesh):
    for p, s in labels.items():
        v = np.array( [i in s for i in range(sampleMesh.GetNumberOfCells())], dtype = float)
        addArrayVTK(sampleMesh, v, name = p, domain = 'cell')
        
    labels['LVepi'] = ((np.where((numpy_support.vtk_to_numpy(sampleMesh.GetCellData().GetArray('LVMyo')) \
    -  1 * numpy_support.vtk_to_numpy(sampleMesh.GetCellData().GetArray('LeftVentricle')) \
    + 2* numpy_support.vtk_to_numpy(sampleMesh.GetCellData().GetArray('MitralValve')) \
    + 2* numpy_support.vtk_to_numpy(sampleMesh.GetCellData().GetArray('AorticValve'))) != 0))[0])
    
def extract_faces(vtkMesh, faces):
    """
    Extracts the cells that belong to a certain label.
    Returns a vtk polydata
    """
    ids = vtk.vtkIdTypeArray()
    for f in faces:
        ids.InsertNextValue(f)
    selectionNode = vtk.vtkSelectionNode()
    selectionNode.SetFieldType(vtk.vtkSelectionNode.CELL)
    selectionNode.SetContentType(vtk.vtkSelectionNode.INDICES)
    selectionNode.SetSelectionList(ids)

    selection = vtk.vtkSelection()
    selection.AddNode(selectionNode)

    extractSelection = vtk.vtkExtractSelection()
    extractSelection.SetInputData(0, vtkMesh)
    extractSelection.SetInputData(1, selection)
    extractSelection.Update()
    polydataConverter = vtk.vtkDataSetSurfaceFilter()
    polydataConverter.SetInputConnection(extractSelection.GetOutputPort())
    polydataConverter.Update()

    return pyvista.PolyData(polydataConverter.GetOutput())

def procrustes_single_mesh(Y, Yref, partial = True):
    Yref = Yref.reshape((-1, 3))
    mu_Y = np.mean(Yref, axis = 0)
    Yref = Yref - mu_Y
    
    y_3D = Y.reshape((-1, 3))
    y_3D = y_3D - np.mean(y_3D, axis = 0)
    
    R, s1 = scipy.linalg.orthogonal_procrustes(y_3D, Yref)
    y_3D =  (y_3D @ R)
    if not partial:
        s =  np.dot(R.flatten(),(y_3D.flatten().T @ Yref).flatten())/ np.linalg.norm(y_3D)**2
        y_3D *= s
    return (y_3D + mu_Y).flatten()

procrustesToMean = lambda Y, Yref, partial = True:np.array([procrustes_single_mesh(y, Yref, partial) for y in Y])

def procrustes(Y, nIts = 20, partial = True):
    #Registers every element to the mean, iteratively until a fix point
    Y_registered = np.copy(Y)
    n = Y_registered.shape[0]
    Y_old = np.zeros(Y.shape[1])
    scales = np.ones(len(Y))
    for i in range(nIts):
        Y_mean = Y[0] if i == 0  else np.mean(Y_registered, axis = 0)
        if np.linalg.norm(Y_old - Y_mean)/np.linalg.norm(Y_mean) < 1e-6:
            break
        else:
            Y_old = Y_mean
        Y_mean_3d = Y_mean.reshape((-1, 3))
        Y_mean_3d = Y_mean_3d - np.mean(Y_mean_3d, axis = 0)
        for i in range(n):
            y_3D = Y_registered[i].reshape((-1, 3))
            y_3D = y_3D - np.mean(y_3D, axis = 0)
            R, s1 = scipy.linalg.orthogonal_procrustes(y_3D, Y_mean_3d)
            Y_registered[i] = (y_3D @ R).flatten()
            if not partial:
                s =   np.dot(R.flatten(),(y_3D.T @ Y_mean_3d).flatten())/ np.linalg.norm(y_3D)**2
                Y_registered[i] *= s
                scales[i] /= s
            else:
                scales[i] = 1
    return Y_registered, scales

def plotModeVariation(mean, mode, var, refMesh, n = 10, path = 'mode'):
    refMesh = pyvista.PolyData(refMesh)
    for i in range(2 * n):
        points = (mean + np.sqrt(var) * (2 * i - n) / n * mode).reshape((-1, 3))
        pathName = str(path) + '_%03d.vtk' % i
        pyvista.PolyData(points, refMesh.faces).save(pathName)
def isWatertight(mesh):
    alg = vtk.vtkFeatureEdges()
    alg.FeatureEdgesOff()
    alg.BoundaryEdgesOn()
    alg.NonManifoldEdgesOn()
    alg.SetInputDataObject(mesh)
    alg.Update()
    is_water_tight = alg.GetOutput().GetNumberOfCells() < 1
    return is_water_tight

def cleanMesh(mesh):
    faces = mesh.faces.reshape((-1, 4))[:, 1:]
    vclean, fclean = _meshfix.clean_from_arrays(mesh.points, faces)
    fclean = np.hstack([3 * np.ones((len(fclean), 1)), fclean]).astype(int)
    mesh = pyvista.PolyData(vclean, fclean)
    return mesh

## Compute the volume of a subpart

class EdgeLocator:
    """
    Dictionary to obtain all faces incident in an edge. Returns the index.
    Warning: assumes an orientable triangular mesh. IE, only at most 2 faces share the same edge, and the are traversed in each direction.
    """ 
    def __init__(self, faces):
        self.edges = collections.defaultdict(list)
        for i,t  in enumerate(faces):
            self.add_edge(t[0], t[1], i)
            self.add_edge(t[1], t[2], i)
            self.add_edge(t[2], t[0], i)

    def add_edge(self, a, b, f):
        if b > a:
            a, b= b, a
        self.edges[(a, b)].append(f)
 
    def getFacesByEdge(self, a, b):
        if b > a:
            a, b= b, a
        return self.edges[(a, b)]

def consistentOrientation(f1, f2):
    """
    Given two triangles, with exactly an intersecting edge, returns True if they have a consistent orientation False otherwise
    """
    edgesF1 = [(f1[0], f1[1]), (f1[1], f1[2]), (f1[2], f1[0])]
    edgesF2 = [(f2[0], f2[1]), (f2[1], f2[2]), (f2[2], f2[0])]
    return not any([e in edgesF2 for e in edgesF1])
        
faceToProcess = collections.namedtuple('FaceToProcess', 'idx needsInvert idxParent')
class TriangleList:
    def __init__(self, trianglesOldMesh, trianglesChoosen = None):
        self.toProcess = set()
        self.trianglesOldMesh = trianglesOldMesh
        self.trianglesChoosen = trianglesChoosen
        self.includedFaces = set()
        
    def is_empty(self):
        return len(self.toProcess) == 0
    
    def pop(self):
        return self.toProcess.pop()
    
    def addFirst(self, t):
        self.toProcess.add(faceToProcess(t, False, -1))
        self.includedFaces.add(t)

    def add(self, t, parentFace, parentFaceId = -1):
        if t in self.includedFaces:
            return
        if self.trianglesChoosen is not None and t not in self.trianglesChoosen:
            return
    
        needsInvert = not consistentOrientation(self.trianglesOldMesh[t] , parentFace)
        self.toProcess.add(faceToProcess(t, needsInvert, parentFaceId))
        self.includedFaces.add(t)
def det3(a):
    return (a[0][0] * (a[1][1] * a[2][2] - a[2][1] * a[1][2])
           -a[1][0] * (a[0][1] * a[2][2] - a[2][1] * a[0][2])
           +a[2][0] * (a[0][1] * a[1][2] - a[1][1] * a[0][2]))

def volumeSubpart(points, triangles):
    acumVol = 0
    mean = np.mean(points, axis = 0)
    for t in triangles:
        acumVol += det3(points[t] )/6
    return np.abs(acumVol)

def getTrianglesSubpartGoodOrientation(trianglesOriginal, trianglesChoosen):
    if len(trianglesChoosen) == 0:
        print('meow')
        return np.zeros((0,3) , dtype = np.int32)
    # Build face adjacency
    e = EdgeLocator(trianglesOriginal)
    faceNeighbours = {}
    for idx, f in enumerate(trianglesOriginal):
        s = set()
        for i, j in [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]:
            for ff in e.getFacesByEdge(i, j):
                s.add(ff)
        faceNeighbours[idx] = s
    
    triangleList = TriangleList(trianglesOriginal, trianglesChoosen)
    triangleList.addFirst(trianglesChoosen[0])
    trianglesProcessed = []
    while not triangleList.is_empty():
        t = triangleList.pop()
        trianglesProcessed.append(trianglesOriginal[t.idx].copy())
        if t.needsInvert:
            trianglesProcessed[-1][0], trianglesProcessed[-1][1] = trianglesProcessed[-1][1], trianglesProcessed[-1][0]
        #else:
        #    print('No invert')
        for neighbour in faceNeighbours[t.idx]:
            triangleList.add(neighbour, trianglesProcessed[-1], t.idx)
    return np.array(trianglesProcessed, dtype = np.int32)