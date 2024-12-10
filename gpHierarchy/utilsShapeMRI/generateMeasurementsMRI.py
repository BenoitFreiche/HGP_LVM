import pyximport; pyximport.install(reload_support=True)
from . import  meshVolumeCython, meshUtils
import numpy as np, pyvista, pandas


def getLA(Y_3D, labels):
    apexPos = np.mean(Y_3D[labels['ApexLVEndo']], axis = 0)
    mitralMean = np.mean(Y_3D[labels['MitralValve']], axis = 0)
    la = mitralMean - apexPos
    
    return np.linalg.norm(la)

def getRVEDV(Y_3D, labelsBiventricular):
    rvedv = meshUtils.cleanMesh(meshUtils.extract_faces(meshSample2, labelsBiventricular['RightVentricle'])).volume
    return (rvedv)/1e6


def getLVEDV(Y_3D, labelsBiventricular):
    lvedv = meshUtils.cleanMesh(meshUtils.extract_faces(meshSample2, labelsBiventricular['LeftVentricle'])).volume
    return (lvedv)/1e6

def getLVM(Y_3D, labelsBiventricular):
    if 'LVepi' not in labelsBiventricular:
        labelsBiventricular['LVepi'] = [i for i in labelsBiventricular['LVMyo'] if i not in  labelsBiventricular['LeftVentricle'] ] + \
                                        [i for i in labelsBiventricular['LeftVentricle'] if i not in  labelsBiventricular['LVMyo'] ] 
    lvepi = meshUtils.cleanMesh(meshUtils.extract_faces(meshSample2, labelsBiventricular['LVepi'])).volume 
    lvedv = meshUtils.cleanMesh(meshUtils.extract_faces(meshSample2, labelsBiventricular['LeftVentricle'])).volume
    return (lvepi - lvedv)/1e6

    


def generateMeasurements(meshesNP, meshSample,  labelsBiventricular = None):
    numpy_to_vtk = lambda x, refMesh = meshSample: pyvista.PolyData(x.reshape((-1, 3)) ,  refMesh.faces)
    lvedv = []
    rvedv = []
    lvm = []
    la = []
    if isinstance(meshesNP, dict):
        keys = meshesNP.keys()
        meshesNP = meshesNP.values()
    else:
        keys = None
        
    pathLabels = './labelsForVolumeSubpart.npz'
    if labelsBiventricular is None:
        
        if pathlib.Path(pathLabels).exists():
            allTriangles = np.load(pathLabels)
            trianglesEpi = allTriangles['trianglesEpi']
            trianglesLV  = allTriangles['trianglesLV']
            trianglesRV  = allTriangles['trianglesRV']
        else:
            raise ValueError()

    else:
        triangles = meshSample.faces.reshape((-1, 4))[:, 1:]
        labelsBiventricular['LVepi'] = [i for i in labelsBiventricular['LVMyo'] if i not in  labelsBiventricular['LeftVentricle'] ] + \
                                [i for i in labelsBiventricular['LeftVentricle'] if i not in  labelsBiventricular['LVMyo'] ] 

        trianglesEpi = meshUtils.getTrianglesSubpartGoodOrientation(triangles.copy(), labelsBiventricular['LVepi'])
        trianglesLV = meshUtils.getTrianglesSubpartGoodOrientation(triangles.copy(), labelsBiventricular['LeftVentricle'])
        trianglesRV = meshUtils.getTrianglesSubpartGoodOrientation(triangles.copy(), labelsBiventricular['RightVentricle'])
        np.savez(pathLabels, trianglesEpi = trianglesEpi, trianglesLV = trianglesLV, trianglesRV = trianglesRV)
        
        
    labelsBiventricularPointwise = meshUtils.cellToPointLabel(labelsBiventricular, meshSample.faces.reshape((-1, 4))[:, 1:])

    for  m in meshesNP:
        #meshSample2 = numpy_to_vtk(m)
        points = m.reshape((-1, 3))
        
        # LV
        #lv_edv = meshUtils.cleanMesh(meshUtils.extract_faces(meshSample2, labelsBiventricular['LeftVentricle'])).volume
        lv_edv = meshVolumeCython.volumeSubpart(points, trianglesLV )

        # RV
        #rv_edv = meshUtils.cleanMesh(meshUtils.extract_faces(meshSample2, labelsBiventricular[ 'RightVentricle'])).volume
        rv_edv = meshVolumeCython.volumeSubpart(points, trianglesRV )

        # LVM 
        #epi =  meshUtils.cleanMesh(meshUtils.extract_faces(meshSample2, labelsBiventricular['LVepi'])).volume
        epi = meshVolumeCython.volumeSubpart(points, trianglesEpi )
        
        lvm_sample = epi - lv_edv
        if lvm_sample < 0:
            meshUtils.extract_faces(points, labelsBiventricular['LVepi']).save('epi.vtk')
            break
            
        # LA 
        la_sample = getLA(points, labelsBiventricularPointwise)
        
        lvedv.append(lv_edv/1e3)
        rvedv.append(rv_edv/1e3)
        lvm.append( lvm_sample/1e3 )
        la.append(la_sample)
    lvm = np.array(lvm)
    rvedv = np.array(rvedv)
    lvedv =  np.array(lvedv)
    df = pandas.DataFrame.from_dict({'lvm': lvm, 'lvedv' : lvedv, 'rvedv' : rvedv, 'la' : la})
    if keys:
        df.index =keys
    return df