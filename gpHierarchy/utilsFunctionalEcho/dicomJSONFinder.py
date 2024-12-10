import pydicom, logging, numpy as np, collections, pathlib, json, os
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

def isSpectralDoppler(self, path, fullRead = False):
    d = pydicom.read_file(str(path), stop_before_pixels= not fullRead)
    if 'SequenceOfUltrasoundRegions' in d:
        for region in d.SequenceOfUltrasoundRegions:
            if region.RegionDataType in [3, 4]: #If region is PW or CW Doppler.
                return True
    return False

class DicomFolder:
    """
    Class for searching a dicom that matches with some characteristics
    """
    def __init__(self, path, fullRead = False):
        """
        fullRead : do not stop before pixels, makes lecture slower. In practice I never foudn that I missed information by stopping early. It takes a while...
        
        Constructs a dict with the info, so each dicom only needs to be read once
        """
        self.patients = collections.defaultdict(dict)
        self.errors = []
        allProcessed = set()
        logging.info('Processing ... It might take a while')
        for p in pathlib.Path(path).glob("**/*"):
            logging.info('Processing ' + str(p))
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
                'rows' : d.Rows,
                'isSpectralDoppler' : False
            }
            ultrasoundRegions = []
            if 'SequenceOfUltrasoundRegions' in d:
                for region in d.SequenceOfUltrasoundRegions:
                    if region.RegionDataType in [3, 4]: #If region is PW or CW Doppler.
                        self.patients[pId][p.stem]['isSpectralDoppler'] = True
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
    def searchByMetadata(self, pId, imageJSON, require_hr_match = True):
        """
        Fast(-er) query
        """
        found = []
        #Search for all images associated to the patient
        for path, image in self.patients[pId].items():
            metadata = imageJSON['metadata']
            #print(metadata['heart_rate'])
            # Do not actually check, since the HR may refer to the image,and not the cycle
            if require_hr_match and metadata['heart_rate'] != int(image['hr']):
                continue 
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
    
    def associateAllTraces(self, jsonPath, jsonPathNew):
        #Read json
        with open(str(jsonPath), 'r') as f:
            patientInfo = json.load(f)
        
        pId = patientInfo['patient_id']
        if len(patientInfo['analysis']) == 0:
            return
        
        for image in patientInfo['analysis'][0]['images']:
            #Image not existing
            if 'id' not in image or not image['id']:
                continue
                
            #Search
            image['path'] = self.searchByMetadata(pId, image)
            if not image['path']:
                self.errors.append('Not found %s - %s' % (pId, image['type']))
        #Write json
        with open(str(jsonPathNew), 'w') as f:
            json.dump(patientInfo, f)
