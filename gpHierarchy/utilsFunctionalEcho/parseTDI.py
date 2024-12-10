import json, os, sys, pathlib, re,collections
import pydicom, pandas, numpy as np, scipy, scipy.interpolate
import matplotlib.pyplot as plt
from . import strainAnalysisTools


def getNamedControlPoints(image):
    controlPoints = {p['type'] : (p['x'], p['y']) for p in image['curve']['control_points']}
    try:
        del controlPoints['mid_control_point']
    except:
        pass
    
    return controlPoints
def correctDrift(t, x):
    t = (t - t[0])/ (t[-1] - t[0])
    x = x - t * (x[-1] - x[0])
    return x

def removeRepeated(t, x):
    idx = np.argsort(t)
    t = t[idx]
    x = x[idx]
    nonRepeated = t == t
    for i, tt in enumerate(t):
        if i + 1 < len(t) and tt == t[i + 1]:
            nonRepeated[i] = False
    return t[nonRepeated], x[nonRepeated]


def interpolateControlPoints(imageJSON, nPoints = 50, mode = 'monotonic'):
    """
    Interpolates a trace from the control points using a piecewise montonic.
    """
    #Get t and y in pixels
    isControlPoint = lambda s: s not in ['cycle begnning', 'cycle end'] or imageJSON['type'] == 'Aortic valve' or 'TDI' in imageJSON['type']
    t = np.array([p['x'] for p in imageJSON['curve']['control_points'] if isControlPoint(p['type']) ])
    y = np.array([p['y']  for p in imageJSON['curve']['control_points'] if isControlPoint(p['type']) ])
    #get physical dimension
    t = (t - imageJSON['onsets']['onset_1']) * float(imageJSON['metadata']['doopler_region']['spacing_x'])
    y = (y - float(imageJSON['metadata']['doopler_region']['zero_line'])) * float(imageJSON['metadata']['doopler_region']['spacing_y'])

    tEvents = {}
    for p in imageJSON['curve']['control_points'] + list(imageJSON['curve'].get('events' , {}).values()):
        if p['type'] == 'mid_control_point':
            continue
        tEvents[p['type']] =  (p['x'] -imageJSON['onsets']['onset_1']) * float(imageJSON['metadata']['doopler_region']['spacing_x'])
    
    #interpolation.
    newTs = np.linspace(0, np.max(t), nPoints)
    if mode == 'pureInterpolation':
        i = scipy.interpolate.interp1d(t, y, kind='cubic', bounds_error = False, fill_value= 0)
        newYs = i(newTs)
    elif mode == 'monotonic':
        t, y = removeRepeated(t, y)
        i = scipy.interpolate.PchipInterpolator(t, y)
        newYs = i(newTs)
    else:
        spline = scipy.interpolate.splrep(t, y, s = 1**2)
        newYs = scipy.interpolate.splev(newTs, spline)
        
    return newTs, newYs, tEvents

def relError(a, b):
    return np.abs((a - b)/(a + b)) * 2

def getEvent(imageSRC, event):
    cSRC = getNamedControlPoints(imageSRC)
    return (cSRC[event][0] - cSRC['cycle beginning'][0])*float(imageSRC['metadata']['doopler_region']['spacing_x'])


def transferEvent(imageSRC, imageDST, event, newEventName = '', interpolate = True):
    if newEventName is None:
        newEventName = event
    cSRC = getNamedControlPoints(imageSRC)
    cDST = getNamedControlPoints(imageDST)
    if interpolate:
        percI1 = (cSRC[event][0] - cSRC['cycle beginning'][0]) /(cSRC['cycle end'][0] - cSRC['cycle beginning'][0])
        xI2 = percI1 * (cDST['cycle end'][0] - cDST['cycle beginning'][0]) + cDST['cycle beginning'][0]
        hr_error_perc = relError(float(imageSRC['metadata']['heart_rate']), float(imageDST['metadata']['heart_rate']))

    else:
        imageSRC_deltaX = float(imageSRC['metadata']['doopler_region']['spacing_x'])
        imageDST_deltaX = float(imageDST['metadata']['doopler_region']['spacing_x'])
        xI2 = (cSRC[event][0] - cSRC['cycle beginning'][0])*imageSRC_deltaX/imageDST_deltaX  +  cDST['cycle beginning'][0]
        hr_error_perc = 0
        
    if 'events' not in imageDST['curve']:
        imageDST['curve']['events'] = {}
    imageDST['curve']['events'][newEventName] = {'type' : newEventName , 'x' : xI2, 'hr_error_perc' : hr_error_perc}

def parseDopplerFile(p, delim = ','):
    """
    Reading the files generated from the plattform, with the curve already processed.
    """
    r = collections.defaultdict(dict)
    with open(p) as f:
        for l in f.readlines():
            if not l or l == '\n':
                continue
            l = l.replace('\n', '').replace('undefined', 'NaN')
            d = l.split(delim)
            r[d[0]][d[1]] = np.array(list(map(float, d[2:])))
    return r
#pathDoppler = pathlib.Path('./Data/doppler/')
#allTraces = {}
#for p in pathDoppler.iterdir():
#    if 'Adol' in str(p) or 'ADUHEART' in str(p):
#        allTraces[p.stem] = parseDopplerFile(p)

def isDoppler(s):
    """
    Returns if the type s corresponds to a doppler image
    """
    return 'PW' in s or 'TDI' in s or 'valve' in s or 'Pulmonary Artery' in s or 'Pulmonary vein' in s
def xToT(x, imageJSON):
    return (x - imageJSON['onsets']['onset_1']) * float(imageJSON['metadata']['doopler_region']['spacing_x'])
def getNamedEvents(imageJSON, acceptedList = ['cycle beginning', 'cycle end', 'valve openning', 'valve closure', 
                                              'diastasis end', 'diastasis beginning', "s'", "e'", "a'", 'AVC', 'MVC', 'MVO', "E", "A"]):
    points = {}
    for c in imageJSON['curve']['control_points']:
        if acceptedList is [] or c['type'] in acceptedList:
            points[c['type']] = {'x' : xToT(c['x'], imageJSON), 'ok' : True}
    if 'events' in imageJSON['curve']:
        for c in imageJSON['curve']['events'].values():
            if acceptedList is [] or c['type'] in acceptedList:
                points[c['type']] = {'x' : xToT(c['x'], imageJSON), 'ok' : float(c['hr_error_perc']) < 0.05}
    return points
def searchImage(jsonFile, type):
    if 'analysis' in jsonFile:
        jsonFile = jsonFile['analysis'][0]
    for i in jsonFile['images']:
        if i['type'] == type and int(i['metadata']['doopler_region']['region_location_min_x']) > 0:
            return i
    return None