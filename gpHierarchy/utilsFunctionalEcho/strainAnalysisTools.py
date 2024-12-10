import os, sys, pathlib
import numpy as np, scipy, pandas
import matplotlib.pyplot as plt
import collections,re, pathlib


###
# Data reading
###
def readSTFile(path):
    floatPattern = '[0-9]*[\.,]?[0-9]+'
    patternFR = f'FR=\s*([0-9]+) Left Marker Time=({floatPattern}) Right Marker Time=({floatPattern}) ES Time=({floatPattern})'
    readNumFrames, readData = False, False
    data = []
    with open(path) as f:
        for l in f.readlines():        
            if re.findall(patternFR, l ): 
                d = re.findall(patternFR, l)[0]
                FR, leftMarkerTime, rightMarkerTime, ES = d[0], d[1], d[2], d[3]

            elif l.startswith('Num Frames:  Knots:'):
                readNumFrames = True

            elif readNumFrames:
                l = l.split(',')[0]
                numFrames, knots = list(map(int, l.split()))
                readNumFrames, readData = False, True
            #Read data
            elif readData:
                line = list(map(float, filter(lambda s: s != '\n', l.split(',') )))
                if len(line):
                    data.append(line)
    xyData = np.array(data).reshape((numFrames, knots, 2))
    return {'KnotData' :xyData, 'FR': int(FR), 'leftMarker' : float(leftMarkerTime), 'rightMarker' :  float(rightMarkerTime), 'ES' : float(ES)}


def getViewFromFilename(filename):
        filename = re.sub( '[^a-zA-Z0-9]', '', filename)
        filename = re.sub( 'FULLTRACE', '', filename)
        if '4CHRV' in filename:
            imageType = '4CH_RV'
        elif '4CHRA' in filename:
            imageType = '4CH_RA'
        elif '4CHLA' in filename:
            imageType = '4CH_LA'
        elif '4CHLV' in filename:
            imageType = '4CH_LV'
        elif 'LV2CH' in filename or '2CHLV' in filename:
            imageType = '2CH_LV'
        elif '2CHLA' in filename:
            imageType = '2CH_LA'
        elif '2CHRV' in filename:
            imageType = '2CH_RV'
        elif '2CHRA' in filename:
            imageType = '2CH_RA'

        elif 'LVAPLAX' in filename:
            imageType = 'PLAX_LV'
        else:
            raise ValueError('Image type unknown', filename)
        return imageType
    
def getPatientId(filename, pattern = 'Adol'):
    return re.findall('%s[0-9]+' % pattern, filename)[0]

def readSpeckleTrackingFromFolder(pathStrainSpeckleTracking, patientIDPattern =  'ADUHEART'):
    if isinstance(pathStrainSpeckleTracking, str):
        pathStrainSpeckleTracking = pathlib.Path(pathStrainSpeckleTracking)
    
    data = collections.defaultdict(dict)
    for p in pathStrainSpeckleTracking.glob('**/*'):
        if p.suffix != '.CSV':
            continue

        imageType = getViewFromFilename(str(p))
        pId = getPatientId(str(p), 'ADUHEART')
        data[pId][imageType] = readSTFile(p)

        # Search  for the file with strain traces (and ECG) associated.
        found = False
        for d in p.parent.glob('*.txt'):
            if pId not in d.name:
                continue
            if any([t not in d.name for t in imageType.split('_')]):
                continue
            found = True
            break
        else:
            print('error', p.name, p)
        if found:
            X = []
            with open(str(d)) as f:
                for i, l in enumerate(f.readlines()):
                    if i >= 4:
                        X.append(list(map(
                                float, l.split())
                                       ))
            X = np.array(X)
            data[pId][imageType]['strainTraces'] = X
            data[pId][imageType]['ECG'] = (X[:, -1])
    return data


#### 
# Computations
####

def getMitralApex(X):
    midMitral = (X[0] + X[-1])/2
    n = X.shape[0]
    apex = (X[n//2] + X[(n + 1)//2])/2
    return midMitral, apex

def getLA(X):
    midMitral, apex = getMitralApex(X)
    la = midMitral - apex
    la = la /np.linalg.norm(la)
    return la

def getPointPerc(Xt, frac = 0.95):
    la = getLA(Xt)
    Xt_prod_la = np.einsum('ni, i->n', Xt, la)
    xt_prod_la_normalised = (Xt_prod_la - np.min(Xt_prod_la))/ (np.max(Xt_prod_la)  - np.min(Xt_prod_la))
    return np.argmax((xt_prod_la_normalised < frac) * len(Xt)**2 + np.arange(len(Xt))) # Trick for getting the index 


def generatePseudoTime(ts, tevent, newtevent):
    return np.interp(ts,  tevent, newtevent)

def temporalAlineation(tRef, t, x):
    return np.interp(tRef,  t, x)

def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def procrustes(X, Y, orthogonal = True):
    Y_wo_mean = Y - np.mean(Y, axis = 0)
    X_wo_mean = X - np.mean(X, axis = 0)
    if orthogonal:
        R, lambdas = scipy.linalg.orthogonal_procrustes(Y_wo_mean,X_wo_mean)
    else:
        raise NotImplemented('Similarity not implemented')
    Y_rotated = np.einsum('ij, ni-> nj', R, Y_wo_mean)
    if not orthogonal:
        pass
    return Y_rotated + np.mean(X, axis = 0)

def generalised_procrustes(X):
    X_mean_old = np.zeros_like(X[0].shape)
    X_mean = X[0]
    second_argument = lambda x, y, z: y
    while np.linalg.norm(X_mean_old - X_mean) > 1e-9:
        #X = [second_argument(*scipy.spatial.procrustes(X_mean, x)) for x in X]
        X = [procrustes(X_mean, x) for x in X]
        X_mean_old = X_mean
        X_mean = np.mean(X, axis = 0)
    return np.array(X)
    

def correctDrift(t, x):
    t = (t - t[0])/ (t[-1] - t[0])
    x = x - t * (x[-1] - x[0])
    return x

# Do the processing
def computeStrain(xyData, ecgEvents, FR,  driftCorrection = False, correctCenter = False,flip = False , is4CH = True, strainComputationType = 'Nico'):
    xyData = xyData.copy()
    if driftCorrection:
        t = (np.arange(xyData.shape[0]) -  ecgEvents['start'])/  (ecgEvents['end'] -  ecgEvents['start'])
        xyData = xyData - np.einsum('i, jk->ijk', t, (xyData[ecgEvents['end']] - xyData[ecgEvents['start']] ))
        
    if correctCenter:
        if is4CH:
            #Correct according the center line.  Note GB: Not sure if it is removing information... It is assuming  a bit of things (ie, y position being approx long)
            # and that the correspondence is OKish. Not sure it is correct.
            n = xyData.shape[1]
            idxL = np.arange(0, n//2 + n%2)
            idxR = np.arange(n-1 , n//2 - 1, step = -1)
            centerLine = (xyData[:, idxL, :] + xyData[:, idxR, :])/2 
            centerLine[:, :, 1] = 0
            print(idxL[: n//2 + n%2].shape)
            xyData[: ,idxL[: n//2], :] -= centerLine[:, : n//2, :]
            xyData[:, idxR, :] -= centerLine
        else:
            xyData -=  p.mean(xyData, axis = 0)
            
    # Anatomical coordinates Compute radial / orthogonal directions
    u_rho = np.zeros([ xyData.shape[0], xyData.shape[1], 2, 2])
    S_longRO = np.zeros([xyData.shape[0], xyData.shape[1]])
    v_rho = np.zeros([ xyData.shape[0], xyData.shape[1], 2])
    exteriorProd = np.array([[0, -1], [1, 0]]) if not flip else np.array([[0, 1], [-1, 0]])
    for i in range(xyData.shape[1]):
        if i == 0:
            if is4CH:
                uo = xyData[:, 1, :] - xyData[:, 0, :]
            else:
                uo = xyData[:, 1, :] - xyData[:, -1, :]
                
        elif i == (xyData.shape[1] - 1):
            if is4CH:
                uo = xyData[:, -1, :] - xyData[:, -2, :]
            else:
                uo = xyData[:, 0, :] - xyData[:, -2, :]

        else:
            uo = xyData[:, i + 1, :] - xyData[:, i - 1, :]
        v_rho[:, i, :] = uo 
        S_longRO[:, i] = np.linalg.norm(uo, axis = 1)
        uo /= np.linalg.norm(uo, axis =1).reshape((-1, 1))
        ur = np.einsum('ij, nj-> ni', exteriorProd, uo)
        
        if is4CH and i > xyData.shape[-1]//2:
            uo *= -1
        u_rho[:, i, 0, :] = uo
        u_rho[:, i, 1, :] = ur
        
    # Strain using the orientation of tn 
    if strainComputationType == 'Nico':
        S_longRO /= np.einsum('tnj, nj -> tn' , u_rho[:, :,0, :], v_rho[ ecgEvents['start'], :, :])
        
    # Strain using 
    elif strainComputationType == 'Length':
        S_longRO /= S_longRO[ecgEvents['start']]
    # Strain using the orientation of t0
    elif strainComputationType == 'Eulerian':
        S_longRO =  np.einsum('nj, tnj -> tn' , u_rho[ecgEvents['start'], :, 0, :], v_rho) / \
                   np.einsum('nj, nj -> n' , u_rho[ ecgEvents['start'], :, 0, :], v_rho[ ecgEvents['start'], :, :])
    
    else:
        raise ValueError('Unkown strain computatino type')
    
    # Displacements in Cartesian
    uxy = xyData - xyData[ecgEvents['start'], :, :].reshape([1, xyData.shape[1], xyData.shape[2]])
    print(uxy.shape, u_rho.shape)
    Phi_rho =  np.einsum('tnij, tnj-> tni', u_rho, uxy)

    # Velocity
    vxy = np.diff(uxy) * FR
    v_rho =  np.einsum('tnij, tnj-> tni', u_rho, vxy)    
    
    #Results
    result = {'xy': xyData, 'Phi_rho': Phi_rho, 'uxy' : uxy, 'strain' : S_longRO}
    return result