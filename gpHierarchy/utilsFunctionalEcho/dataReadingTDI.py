from . import strainAnalysisTools
import pathlib,json, collections, tqdm, numpy as np, logging, collections

def readDicomImage(p):
    d = pydicom.read_file(p)
    rgb = pydicom.pixel_data_handlers.util.convert_color_space(d.pixel_array,  d.PhotometricInterpretation, 'RGB')
    return rgb, d

def selectDopplerRegion(d):
    pass

def readJSON(p):
    with open(str(p), 'r') as f:
        r = json.load(f)
    return r

def inferType(jsonFile):
    if 'jsonType' in jsonFile:
        return jsonFile['jsonType']
    else:
        return 'Carlos' if 'metadata' in jsonFile['images'][0] else 'Processed'
def searchImage(jsonFile, type):
    if 'analysis' in jsonFile:
        jsonFile = jsonFile['analysis'][0]
    typeJSON = inferType(jsonFile)

    for i in jsonFile['images']:
        # If Carlos'
        if typeJSON == 'Carlos' and i['type'] == type and int(i['metadata']['doopler_region']['region_location_min_x']) > 0:
            return i
        #If it is as the ones that I created
        elif typeJSON == 'Processed' and i['name'] == type:
            return i
    return None

def listImages(jsonFile):
    if 'analysis' in jsonFile:
        jsonFile = jsonFile['analysis'][0]
    typeJSON = inferType(jsonFile)

    images = []
    for i in jsonFile['images']:
        # If Carlos'
        if typeJSON == 'Carlos' and  int(i['metadata']['doopler_region']['region_location_min_x']) > 0:
            images.append(i['type'])
        #If it is as the ones that I created
        elif typeJSON == 'Processed':
            images.append(i['name'])

    return images

def readDataTracesFromJSON(dataToRead, path, filterConstantFeatures = True, onlySystole = False, selectPatient = lambda s: 'ADUHEART' in s):
    """
    dataToRead: dictionary 
    """
    # Read the data
    results = {}
    for name, info in dataToRead.items():
        d =  TraceDataset()
        n = info.get('signalName', name)
        temporalEvents = info.get('temporalEvents', [])
        _ = d.generateDataset(path, n, temporalEvents, selectPatient =selectPatient)
        results[name] = d
        print(d.events)
    # Join all the datasets
    resultsJoined = {}
    for name, d in results.items():
        dJoined = d
        for name2, d2 in results.items():
            if name2 != name:
                dJoined = dJoined.joinDatasets(d2)
        dJoined.ys = np.array(dJoined.ys)
        dJoined.tsOriginal = np.array(dJoined.tsOriginal)
        dJoined.ts = np.array(dJoined.ts)

        #Select systole. We can do this because MVO is a registered event.
        if onlySystole:
            try:
                mvo = dJoined.events['MVO']
                dJoined.tsOriginal = dJoined.tsOriginal[:, :mvo]
                dJoined.ys = dJoined.ys[:, :mvo]
            
            except:
                logging.info(f'Could not find MVO in trace: {name}. Could not select the systole.')
            
        #Remove the features that are constant (ie, as when the valve is closed and there is no flow in flow Doppler)
        if filterConstantFeatures:
            std = np.nanstd(dJoined.ys, axis = 0)
            idx = np.where(std > 1e-6)[0]
            dJoined.ys = dJoined.ys[:, idx]
            dJoined.tsOriginal = dJoined.tsOriginal[:, idx]
            dJoined.ts = dJoined.ts[:, idx]
        Y = dJoined.ys
        
        dJoined.computeCentered()
        resultsJoined[name] = dJoined

    return resultsJoined



class DicomFolder:
    """
    Class for searching a dicom that matches with some characteristics
    """
    def __init__(self, path, fullRead = False):
        """
        fullRead : do not stop before pixels, makes lecture slower. In practice I never foudn that I missed information by stopping
        
        Constructs the info
        """
        self.patients = collections.defaultdict(dict)
        allProcessed = set()
        for p in tqdm.notebook.tqdm(pathlib.Path(path).glob("**/*")):
            print(p)
            # Do not consider hidden files, or already processed continue
            if p.stem.startswith('.') or  p.stem in allProcessed:
                continue
            
            #If it is not a dicom, ignore
            try:
                d = pydicom.read_file(str(p), stop_before_pixels= not fullRead)
            except Exception as e:
                continue
            pId = d.PatientID
            self.patients[pId][p.stem] = {
                'path' : str(p),
                'hr' :  np.nan if 'HeartRate' not in d else d.HeartRate,
                'cols' : d.Columns,
                'rows' : d.Rows
            }
            ultrasoundRegions = []
            if 'SequenceOfUltrasoundRegions' in d:
                for region in d.SequenceOfUltrasoundRegions:
                    try:
                        ultrasoundRegions.append({
                            'minx' :region.RegionLocationMinX0,
                            'maxx' :region.RegionLocationMaxX1,
                            'miny' :region.RegionLocationMinY0,
                            'maxy' :region.RegionLocationMaxY1,
                            'deltaX' : region.PhysicalDeltaX,
                            'deltaY' : region.PhysicalDeltaY,
                            'zero_line' : region.ReferencePixelY0

                        })
                    except:
                        pass
            self.patients[pId][p.stem]['regions'] = ultrasoundRegions
            allProcessed.add(p.stem)

            
    #Check if there is a compatible 
    def searchByMetadata(self, pId, json):
        """
        Fast(-er) query
        """
        found = []
        #Search for all images associated to the patient
        for path, image in self.patients[pId].items():
            metadata = imageJSON['metadata']
            #print(metadata['heart_rate'])
            # Do not actually check, since the HR may refer to the image,and not the cycle
            #if metadata['heart_rate'] != int(image['hr']):
            #    continue 
            if (metadata['image_size']['width'] != image['cols'] or 
               metadata['image_size']['height'] != image['rows']):
                continue
            #Search for all regions
            regionLocationMin_x, regionLocationMax_x, regionLocationMin_y, regionLocationMax_y = metadata['doopler_region']['region_location_min_x'], metadata['doopler_region']['region_location_max_x'], metadata['doopler_region']['region_location_min_y'], metadata['doopler_region']['region_location_max_y']

            for p in image['regions']:
                    regionOK = (p['minx']  == int(regionLocationMin_x)) and  \
                                                       (p['maxx'] == int(regionLocationMax_x)) and \
                                                     (p['miny'] == int(regionLocationMin_y)) and (p['maxy'] == int(regionLocationMax_y))
                    unitsOK = np.isclose(float(metadata['doopler_region']['spacing_x']), p['deltaX']) and np.isclose(p['deltaY'], float(metadata['doopler_region']['spacing_y']))
                    referencePixelOK = int(metadata['doopler_region']['zero_line']) == p['zero_line']
                    if regionOK and unitsOK and referencePixelOK:
                        found.append(image['path'])
                        break
        return found
    
    

def zero_crossing(X):
    return np.where(X[1:] * X[:-1] < 0)[0]
    
def selectDiastoleZeroCrossing(record, X, Y, event = "s'", nSamples = 100):
    """
    Selects the diastole as the first time that the TDI trace reaches 0 before s'
    """
    zeros = zero_crossing(Y)
    s_prime_x = record['events'][event]['x']
    for i in zeros:
        if X[i] > s_prime_x:
            break
    else:
        raise ValueError('Could not find 0 crossing')
    X = X[i:] 
    Y = Y[i:]
    t = np.arange(X.shape[0])
    tRef = np.linspace(0, X.shape[0], nSamples)
    X = np.interp(tRef,  t, X)
    Y= np.interp(tRef,  t, Y)
    return X, Y 
    
class AlignedTraceDataset:
    """
    Assumes that traces are already interpolated (done in the previous step when generating the JSON)
    
    """
    def generateDataset(self, folder, imageName,  ecgEvents = [], 
                        driftCorrection = True, invert = True, interpolate = None, selectDiastole = False, 
                        selectPatient = lambda s: 'Adol' in s, numPoints = 150):
        """
        Generates a dataset ready to use by the algorithms from the JSON files.
        """
        self.ts, self.ys, self.names, self.tsOriginal = [], [], [], []
        tRef = np.linspace(0, len(ecgEvents) + 1, num = numPoints)
        # TODO : Sort the ecg events
        self.events =  None #{e : np.argmin(np.abs(tRef - i - 1)) for i, e in enumerate(ecgEvents)}
        for p in pathlib.Path(folder).iterdir():
            # Read data
            if p.suffix != '.json' or not selectPatient(str(p)):
                continue
            
            pId = p.stem
            with open(p) as f:
                d = json.load(f)
                
            if 'images' not in d:
                print('No images found! Are you sure you preprocessed the JSONs?')

            i = d['images'].get(imageName, None)
            if i is None:
                print(imageName, 'Not found on patient', pId)
                continue
                
            # Check that all events are OK
            try:
                if not all([ i['events'][e]['ok'] for e in ecgEvents]):
                    print('Event unreliable - ignore')
                    continue
            except:
                continue
            #
            t = i['traces']['t']
            y =  i['traces']['velocity']
            
            #Cut cycle beginning / cycle ending
            t = np.array(t)
            tBegin = i['events']['cycle beginning']['x']
            idx = np.where(np.logical_and(t > i['events']['cycle beginning']['x'], t < i['events']['cycle end']['x']))[0]
            t = [t[i] - tBegin for i in idx]
            y = [y[i] for i in idx]
            
            if invert:
                y = [-yy for yy in y]
            # They should always be in order!
            eventsTs = np.sort([ i['events'][e]['x'] for e in ecgEvents])
            if self.events is None:
                idxEvents =  np.argsort([ i['events'][e]['x'] for e in ecgEvents])
                self.events = {ecgEvents[i] : np.argmin(np.abs(tRef - j - 1)) for j, i in enumerate(idxEvents)}
                
            # Will reduce a bit the extremes...
            if interpolate:
                i1 = np.linspace(0, 1, num = len(t))
                i2  = np.linspace(0, 1, num = interpolate)
                t = np.interpolate(i1, i2, t)
                y = np.interpolate(i1, i2, y)
            if driftCorrection:
                y = strainAnalysisTools.correctDrift(np.array(t), np.array(y))
                
            if selectDiastole:
                t, y = selectDiastoleZeroCrossing(i, np.array(t), np.array(y))
            
            #Temporal alignment
            if ecgEvents:                
                if len(self.ys) == 0:
                    tRef = t
                    tRefSynthetic =  strainAnalysisTools.generatePseudoTime(t, 
                                                                                  np.concatenate([[0], eventsTs,  [t[-1]]]), list(range(len(ecgEvents) +2)))
                    idxEvents =  np.argsort([ i['events'][e]['x'] for e in ecgEvents])
                    self.events = {ecgEvents[i] : np.argmin(np.abs(tRefSynthetic - j - 1)) for j, i in enumerate(idxEvents)}

                tSynthetic = strainAnalysisTools.generatePseudoTime(t, np.concatenate([[0], eventsTs,  [t[-1]]]), list(range(len(ecgEvents) +2)))
                
                yResampled = strainAnalysisTools.temporalAlineation(tRefSynthetic, tSynthetic, y)
                tOriginal =  strainAnalysisTools.temporalAlineation(tRefSynthetic, tSynthetic, t)
                #tResampled = strainAnalysisTools.temporalAlineation(tRefSynthetic, tSynthetic, t)
                tResampled = tRef
            else:
                tResampled, yResampled = t, y
                tOriginal = t
            self.ts.append(tResampled)
            self.tsOriginal.append(tOriginal)
            self.ys.append(yResampled)
            self.names.append(pId)
        self.ts, self.ys, self.names = np.array(self.ts), np.array(self.ys), np.array(self.names)
        
        resultDoppler = collections.namedtuple('ResultDoppler', 'Time Speed Names Events')
        return resultDoppler(self.ts, self.ys, self.names, self.events)
    
    
    def copy(self):
        other = TraceDataset()
        other.ts = self.ts.copy()
        other.tsOriginal = self.tsOriginal.copy()
        other.ys = self.ys.copy()
        other.names = self.names.copy()
        other.events = self.events
        return other
    
    def sortByName(self):
        """
        Sort,
        TODO: use property to have registered variables and do not add here one by one
        """
        idxSorted = sorted(range(len(self.ts)), key = lambda i: self.names[i])
        self.ts = np.array([self.ts[i] for i in idxSorted])
        self.ys = np.array([self.ys[i] for i in idxSorted])
        self.names = np.array([self.names[i] for i in idxSorted])
        self.tsOriginal = np.array([self.tsOriginal[i] for i in idxSorted])
        
    def select(self, idx):
        self.ts = self.ts[idx]
        self.ys = self.ys[idx]
        self.tsOriginal = self.tsOriginal[idx]
        self.names = self.names[idx]
        self.computeCentered()
        
    def computeCentered(self):
        self.tsCentered = self.tsOriginal - np.nanmean(self.tsOriginal, axis = 0).reshape((1, -1))
        self.Ycentered = self.ys - np.nanmean(self.ys, axis = 0).reshape((1, -1))
        self.Yt = np.concatenate([self.ys/np.nanmean(np.nanstd(self.ys, axis = 0)),  self.tsOriginal/np.nanmean(np.nanstd(self.ys, axis = 0))], axis = 1)


    def joinDatasets(self, other, fillValue = np.nan):
        """
        Add nans for non - complete data
        """
        s = self.copy()
        nTemporalSamples = len(s.ts[0])
        for i, n in enumerate(other.names):
            

            if n in self.names:
                #Already in the dataset, nothing needed
                continue
            #If it is not included, add a nan  opy
            s.ts = np.append(s.ts, np.repeat(fillValue, nTemporalSamples).reshape((1, -1)), axis = 0)
            s.ys = np.append(s.ys, np.repeat(fillValue, nTemporalSamples).reshape((1, -1)), axis = 0)
            s.tsOriginal = np.append(s.tsOriginal, np.repeat(fillValue, nTemporalSamples).reshape((1, -1)), axis = 0)
            s.names = np.append(s.names, other.names[i])
        s.sortByName()
        s.computeCentered()
        return s