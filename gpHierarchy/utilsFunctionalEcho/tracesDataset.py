from . import parseTDI, dataReadingTDI

import os, numpy as np, json, pathlib, re

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def getStrainType( p):
        if 'FULL_TRACE' in str(p):
            isFullTrace = True
        else:
            isFullTrace = False

        fileName = re.sub( '[^a-zA-Z0-9]', '', p.stem)
        if '4CHSL4CHRVTRACE' in fileName or '4CHSLRVTRACE' in fileName:
            imageType = '4CH_RV'
        elif '4CHSL4CHRATRACE' in fileName or '4CHSLRA' in fileName:
            imageType = '4CH_RA'
        elif '4CHSL4CHLATRACE' in fileName or '4CHSLLA' in fileName:
            imageType = '4CH_LA'
        elif '4CHSLLVTRACE' in fileName:
            imageType = '4CH_LV'
        elif 'SLLV2CHTRACE' in fileName or '2CHSLLVTRACE' in fileName:
            imageType = '2CH_LV'
        elif 'SL2CHLATRACE' in fileName or '2CHSLRA' in fileName:
            imageType = '2CH_LA'
        elif 'SLLVAPLAXTRACE' in fileName:
            imageType = 'PLAX_LV'
        else:
            print('Type of ', p , ' not recognised - ignoring')
            imageType = None
        return imageType, isFullTrace
    
class DatasetTracesJSON:
    def __init__(self):
        self.patients = {}
        
    def loadTDI(self, pathTDI):
        allImages = {}
        with open(str(pathTDI)) as f:
            d = json.load(f)
        if not d['analysis']:
            return
        
        aortaImage = parseTDI.searchImage(d, 'Aortic valve')
        mitralImage = parseTDI.searchImage(d,'Mitral valve PW Doppler')
        pId = d['patient_id'] 

        for imageJSON in d['analysis'][0]['images']:
            metadata = imageJSON['metadata']
            if not parseTDI.isDoppler(imageJSON['type']):
                continue

            # If the trace is not missing to start with
            if  metadata['image_size']['width'] <= 0: 
                continue

            points = []
            for t in imageJSON['curve']['control_points']:
                # DUnno why I don't have to add offset in X, but seemingly works
                # They use interpolation
                if t['x'] < 0:
                    continue
                points.append((t['x'] , 
                               t['y'] + int(metadata['doopler_region']['region_location_min_y'])
                              ))
            if len(points) < 2:
                continue

            if 'TDI' in imageJSON['type']:
                # Add Aortic valve clousure event to all TDI traces
                t = imageJSON

                if aortaImage:
                    parseTDI.transferEvent(aortaImage, t, 'valve closure', 'AVC_interpolation')
                    parseTDI.transferEvent(aortaImage, t, 'valve closure', 'AVC', interpolate = False)

                #Mitral valve opening and closing       
                if mitralImage:
                    parseTDI.transferEvent(mitralImage, t, 'valve openning', 'MVO', interpolate = False)
                    parseTDI.transferEvent(mitralImage, t, 'valve closure', 'MVC', interpolate = False)

            points = np.array(points, dtype = int)
            t, v, _ = parseTDI.interpolateControlPoints(imageJSON, 100)
            doppler = {
                'name' : imageJSON['type'],
                'type' : 'Doppler',
                'image' : None,
                'controlPointsX' : list(map(int, points[:, 0])),
                'controlPointsY' : list(map(int, points[:, 1])),
                'X0' : int(points[0,0]),
                'deltaX': float(metadata['doopler_region']['spacing_x']),
                'deltaY': float(metadata['doopler_region']['spacing_y']),
                'y0' : float(metadata['doopler_region']['zero_line']),
                'events' : parseTDI.getNamedEvents(imageJSON),
                'traces' :{
                    't' : list(t),
                    'velocity' : list(v)
                }
            }
            # I won't allow several images of the same type on the same patient
            if doppler['name'] in allImages:
                raise Error('It is not allowed to have repeated sequences.')
            allImages[doppler['name']] = doppler 
        
        dictionary = self.patients.get(pId, {}) 
        dictionary.update(allImages)
        self.patients[pId] = dictionary
        
    def loadStrain(self, pathStrainTrace, pId):
        imageType, knotData =getStrainType(pathStrainTrace)
        if knotData or imageType is None:
            return
        
        with open(pathStrainTrace) as f:
            l = f.readlines()[4:]
            res = []
            for ll in l:
                res.append(list(map(float, (re.sub( '\s+', ' ', ll)).split())))
        res = np.array(res)
        t = list(res[:, 0])
        ecg = list(res[:, -1])
        gls = list(res[:, -2])

        strain = {
            'name' : imageType,
            'type' : 'ST',
            'traces' : {
                't' : t,
                'gls' : gls,
                'ecg' : ecg
                
            },
            'events' : {}
        }
        d = self.patients.get(pId, {})
        d['ST_' + imageType ]  = strain
        self.patients[pId] = d
        
    def save(self, path):
        for p, allImages in self.patients.items():
            d = {'pId' : p,
                'images' : allImages
            }
            with open( os.path.join(path, '%s.json' % p), 'w') as fp:
                json.dump(d, fp, sort_keys=True, indent=4, cls = NumpyEncoder)
