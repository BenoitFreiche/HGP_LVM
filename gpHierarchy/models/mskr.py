"""
From Benoit Freiche, python adaptation from code of N. Duchteau
Reference: https://core.ac.uk/download/pdf/237332624.pdf
"""
import numpy as np
import sklearn, sklearn.model_selection
from sklearn.metrics import DistanceMetric
from sklearn.metrics import pairwise_distances
# from scipy.spatial.distance import pdist as pairwise_distances
from types import SimpleNamespace
from math import pi
import operator
import logging

def nandot(A,B):
    return np.dot(A,B)


#modif => nandot à 3 endroits. c à vérifier mais les résultats paraissent ok

def MINEXACT_Bermanis_ACHA_2013_ALGO_03( B_s , f , x_s , x_star , e_s , gamma , i_list , NYSTROM , usePINV ) :
    
    [n,l] = np.shape(B_s);

    try : 
        np.shape(f)[1] >= 1
    except :
        f = np.reshape(f,(-1,1))

    if( (n == l) & (l == len(i_list)) ):
        EYE = np.eye(n);
    else :

        EYE = np.zeros((n,l))

        for j in range(len(i_list)):

            EYE[i_list[j],j] = 1; 

    

    Ktmp = ( B_s + (1/gamma) * EYE )



    #step 3

    if usePINV == 1 : 

        # slower version for small number of attributes: 

        if n == l:

            B_s_cross = np.linalg.inv( Ktmp );

        else :

            B_s_cross = np.linalg.pinv( Ktmp );

        try :

            nD = np.shape(f)[1]

        except:

            nD = 1

#    c = zeros(l,nD); for i=1:nD; c(:,i) = B_s_cross * f(:,i); end;

        c = np.transpose(nandot(np.transpose(f) ,B_s_cross));

    else:

        try :

            nD = np.shape(f)[1]

        except:

            nD = 1

    #     c = zeros(l,nD); for i=1:nD; c(:,i) = Ktmp \ f(:,i); end;

        c = np.transpose((np.transpose(f) / Ktmp));

    #step 4

    #f_s = np.dot(c.T , B_s).T

    f_s = nandot(c.T , B_s).T

    p = len(x_star);

    if len(np.shape(x_s)) >= 3:

        x_star = np.reshape(x_star,(len(x_star),-1))

        x_s = np.reshape(x_s,(len(x_s),-1))

    tmp2 = pairwise_distances(x_star,x_s);

    

    if NYSTROM == 0 :

        tmp2 = np.exp( -tmp2**2 / e_s );

    else :

        tmp2 = np.exp( -tmp2**2 / (2*e_s**2) )

    #f_star_s = np.dot(c.T , tmp2.T).T

    f_star_s = nandot(c.T , tmp2.T).T

    output = SimpleNamespace()

    output.f_s = f_s

    output.f_star_s = f_star_s

    return output



class MKSRWrapper(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    def __init__(self, startScale = 0, gammaList = [1], kNN =7, densityFactor = 1,
                           useSingleScale = False):
        self.startScale = startScale
        self.gammaList = gammaList
        self.kNN = kNN
        self.densityFactor = densityFactor
        self.useSingleScale = useSingleScale
    
    def fit(self, X, y):
        self.Xtrain = X
        self.meanY = np.mean(y, axis = 0)
        self.Ytrain = y  - self.meanY
    def predict(self, X):
        OUT = interpolateMultiPython(self.Xtrain, self.Ytrain, X, 
                                     startScale = self.startScale, gammaList = self.gammaList, kNN = self.kNN,
                                     densityFactor = self.densityFactor, useSingleScale=  self.useSingleScale).OUT[0]
        return OUT + self.meanY
    
    def crossval_predict(self,X, Y, k = 10 ):
        """
        Warning, only tested for 1 dimensional output
        """
        cv = sklearn.model_selection.KFold(k)
        res = np.zeros(Y.shape)
        idTest = []
        for idtrain, idtest in cv.split(X):
            self.fit(X[idtrain], Y[idtrain])
            res[idtest] = self.predict(X[idtest]).reshape(res[idtest].shape)
        return res       

def interpolateMultiPython(xi, yi, xTest, startScale = 0, gammaList = [1], kNN =7, densityFactor = 1,
                           useSingleScale = False, testIndices = [],sigma = None):
    IN = SimpleNamespace()
    IN.gammaList = gammaList ## here, two regularization weigths tested
    IN.startScale = startScale
    IN.densityFactor = densityFactor
    if sigma:
        IN.sigma = sigma
    
    idx = np.array([i not in testIndices for i in range(len(xi))], dtype = bool)
    idx2 = np.logical_and(xi[:, 0] == xi[:, 0], yi[:, 0] == yi[:, 0])
    idx = np.logical_and(idx, idx2)
    IN.xi = xi[idx]
    IN.yi = yi[idx]
    IN.x = xTest
    IN.kNN = kNN
    return interpolateMulti(IN)


def interpolateMulti(IN):

    data = SimpleNamespace()
    testData = SimpleNamespace()

    data.x = IN.xi
    data.y = IN.yi

    data.numS = len(data.x)
    try : 
        data.dim  = np.shape(data.y)[1]
    except:
        data.dim = 1

    testData.x = IN.x
    testData.numS = len(testData.x)

    

    if hasattr(IN,'kNN'): # number of nearest neighbors used to estimate the density of samples
        kNN = IN.kNN;
    else:
        kNN = 10

        

    if hasattr(IN,'singleScale') : # use a single scale

        useSingleScale = 1

        data.T = IN.singleScale**2

    else :

        useSingleScale = 0

        data.T = np.nan

        

    if hasattr(IN,'densityFactor') : # at which scale to stop the iterations (density * factor)

        densityFactor = IN.densityFactor

    else :

        densityFactor = 1

        

    if hasattr(IN,'startScale') : # at which scale to start

        startScale = IN.startScale

    else : 
        startScale = 0

        

    if hasattr(IN,'gammaList') : # weight(s) between regularization and similarity (the lower the smoother)
        data.gammaList = IN.gammaList

    else :
        data.gammaList = 10**0

    

    if hasattr(IN,'usePINV') : # use pinv for inverse computations
        usePINV = IN.usePINV

    else :
        usePINV = 1

    

    if hasattr(IN,'useNYSTROM') : # use the definition of Nystrom extension
        useNYSTROM = IN.useNYSTROM

    else :
        useNYSTROM = 0

    

    if hasattr(IN,'exactPoints') :
        exactPoints = IN.exactPoint; # list of samples where to have exact matching

    else :
        exactPoints = []

    dist = DistanceMetric.get_metric('euclidean')

    if len(np.shape(data.x)) >= 3:
        reshaped_data = data.x
        reshaped_data = np.reshape(reshaped_data,(len(reshaped_data),-1,1))[:,:,0]
        Dg = dist.pairwise(reshaped_data)
        
    else : 
        Dg = dist.pairwise(data.x)


    # Estimating data density

    data.diam = np.max(Dg)

    tmp = Dg + np.diag([np.inf]*data.numS)  # put diagonal coefficients to -1

    tmpB = np.sort(tmp)

    tmpB = tmpB[:,:min(kNN-1,data.numS-2)] # approximate density from k neighbors

    tmpB = np.mean(tmpB)

    data.density = np.mean(tmpB);


    if useSingleScale == 0 :

        it_max = 20;

        data.T = data.diam**2

    x = data.x

    x_star = testData.x

    f = data.y

    

    OUT = SimpleNamespace()



    OUT.OUT = []

    OUT.finalT = []

    for gI, v in enumerate(data.gammaList) :

        #logging.debug('gI #' + str(gI))
        gammaI = v
        gamma = gammaI;

        F_s_old = np.zeros((data.numS,data.dim));
        F_star_s_old = np.zeros((testData.numS,data.dim));

        

        s = startScale

        #############################

        

        if(useSingleScale == 1) :
            e_s = data.T / 2**s;
            Ge_s = np.exp( -Dg**2 / e_s );  # (n,n)
            pointsInexact = np.array(range(len(Ge_s))) 
            pointsInexact[exactPoints] = []
            output_03 = MINEXACT_Bermanis_ACHA_2013_ALGO_03( Ge_s , f - F_s_old , x , x_star , e_s , gamma , pointsInexact , useNYSTROM , usePINV);
            F_star_s_old = output_03.f_star_s;

        else :

            # Stop when the kernel size is smaller than the density of samples
            while ( ( s <= it_max ) & ( np.sqrt(data.T/2**s) > (densityFactor*data.density) ) ) :
                #logging.debug(str(s) + ' ')
                e_s = data.T / (2**s)
                Ge_s = np.exp( -Dg**2 / e_s ) # (n,n)
                pointsInexact = np.array(range(len(Ge_s)))
                pointsInexact[exactPoints] = [];
                output_03 = MINEXACT_Bermanis_ACHA_2013_ALGO_03( Ge_s , f - F_s_old , x , x_star , e_s , gamma , pointsInexact , useNYSTROM , usePINV);
                F_s = F_s_old + output_03.f_s
                F_star_s = F_star_s_old + output_03.f_star_s
                s+= 1;
                F_s_old = F_s;
                F_star_s_old = F_star_s
            OUT.finalT.append([np.sqrt(data.T/2**(s-1)),s-1])
            
        OUT.OUT.append(F_star_s_old)
        OUT.density = data.density

    OUT.diam = data.diam;
    print('df',densityFactor)
    print('density',data.density)
    print('stratscale : ',startScale)
    return OUT

