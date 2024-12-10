import pickle, pathlib, pyvista
import numpy as np, pandas
import pickle, sys
from . import meshUtils

def readMeshesEDFromVTK(pathMeshes, pathLabels = './Data/aduheartMRI/fullLabelCorrected.lbl', 
                        picklePath = None, recompute = False, sampleMeshSavePath = '', use_ed_r_peak = True):
    """
    Reads the full meshes (4 chambers), selects the ED and removes all but the LV.
    """
    if picklePath is not None and  pathlib.Path(picklePath).exists() and not recompute:
        with open(picklePath, 'rb') as fp:
            meshes = pickle.load(fp)

    else:
        labelsFull = meshUtils.readLabels(pathLabels)
        facesBiventricular = sorted(list(set(labelsFull['LeftVentricle'] + labelsFull['RightVentricle'] + labelsFull['LVMyo'])))
        meshes = {}
        for p in pathlib.Path(pathMeshes).glob('*'):
            if p.name.startswith('.'):
                continue

            # Read and select ED
            # If ED using index
            if use_ed_r_peak:
                mesh_ed = pyvista.PolyData(p / (p.name + '_000.vtk'))
            # If ED using minimum 
            else:
                vol_lv = -np.inf
                for pp in p.glob('*'):
                    mesh = pyvista.PolyData(pp)
                    lv = meshUtils.extract_faces(mesh, labelsFull['LeftVentricle'])
                    if vol_lv < lv.volume:
                        vol_lv = lv.volume
                        mesh_ed = mesh
            # Select the Left and right ventricle
            meshBiventricular =  meshUtils.extract_faces(mesh_ed, facesBiventricular)
            meshes[p.name] = meshBiventricular.points.flatten().astype(float)
            if sampleMeshSavePath:
                meshBiventricular.save(sampleMeshSavePath)
        if picklePath:
            with open(picklePath, 'wb') as fp:
                pickle.dump(meshes, fp)
    return meshes

def generateDataset(meshes, rigidProcrustes, scaling = None):
    Y_ED_original_unscaled = np.array(list(meshes.values()))
    Y_ED_with_scale = Y_ED_original_unscaled.copy()
    names = list(meshes.keys()) 

    Y_ED, scales = meshUtils.procrustes(Y_ED_with_scale, partial = rigidProcrustes)
    if scaling is not None and not rigidProcrustes:
        raise ValueError('It does not make sense to apply an scaling if scale was removed during Procrustes.')
        
    if scaling is not None:
        Y_ED /= scaling
        scales *= scaling
    return Y_ED, scales, names
