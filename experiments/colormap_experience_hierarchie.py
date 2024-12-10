import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

def Flatten(images):
    return [im.flatten() for im in images]

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib as mpl
mpl.rcParams['font.size'] = 20    
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def getScatterImage(vector,shape = (218,178),size = 2):
    x_vector = vector[0::2]
    y_vector = vector[1::2]
    im = np.zeros(shape)
    for i in range(len(x_vector)):
        im[(y_vector[i]-size):(y_vector[i]+size),(x_vector[i]-size):(x_vector[i]+size)] = 255
    im[:,0:2] = 190
    im[:,-2:] = 190
    im[0:2,:] = 190
    im[-2:,:] = 190
    return im
    
def getImage(path,zoom = 0.3,cmap =  mpl.cm.gray, norm=mpl.colors.Normalize(vmin=0, vmax=1)):
    return OffsetImage(path,zoom = zoom,cmap = cmap,norm=norm)

def UnFlatten(flat,shape = (218,178)):
    return [vect.reshape(shape) for vect in flat]

os.chdir('C:/Users/b_freich/Documents/Research/Code/Code_these_pour_ND/data/celebA_brunes')


attr = pd.read_csv('list_attr.csv')
bbox = pd.read_csv('list_bbox.csv')
partition = pd.read_csv('list_eval_partition.csv')
landmarks = pd.read_csv('list_landmarks.csv')

images = []
key_im = []
n_image = 4714
egal = True
if egal:
    dir_ = 'images_egalisees'
else :
    dir_ = 'images'
#outliers = [40,119,198,239,394,416]
seed = 50 #30
np.random.seed(seed)
list_index = np.arange(n_image)
list_names = np.asarray(np.sort(os.listdir(dir_)))[list_index]
#s = []
for image in list_names:
    im_ = plt.imread(os.path.join(dir_,image))[:,:,0]
    images.append(im_.astype(int))   
    key_im.append(image)

boolean = [landmark_key in key_im for landmark_key in landmarks.values[:,0]]

land = landmarks.loc[boolean]
land = land.values[:,1:].astype(float)

at = attr.loc[boolean]
at = at.values[:,1:]

N = 400
data_images = np.asarray((images)[:N])/255
data_reshape = data_images.reshape((400,-1))

land = land[:N].astype(int)

def pos_nose_x(landmark):
    mean_left = (landmark[0]+landmark[6])/2
    mean_right = (landmark[2]+landmark[8])/2
    nose = landmark[4]
    return (nose-mean_left)/(mean_right-mean_left)

def pos_nose_y(landmark):
    mean_left = (landmark[7]+landmark[9])/2
    mean_right = (landmark[3]+landmark[1])/2
    nose = landmark[5]
    return (nose-mean_left)/(mean_right-mean_left)
    
def smile(landmark):
    return landmark[8] - landmark[6]
    

    
M = map(pos_nose_x,land)
color_x = list(M)
color_x = (color_x - np.min(color_x)) / (np.max(color_x )- np.min(color_x))

M = map(pos_nose_y,land)
color_y = list(M)
color_y = (color_y - np.min(color_y)) / (np.max(color_y )- np.min(color_y))

M = map(smile,land)
color_smile = list(M)
color_smile = (color_smile - np.min(color_smile)) / (np.max(color_smile )- np.min(color_smile))

def mean_image(image):
    return np.mean(image)

def droite_1(landmark):
    #definition des droites
    coef = (landmark[1]-landmark[3])/(landmark[0]-landmark[2])
    ordonnee = landmark[1] - coef*landmark[0]
    return coef,ordonnee
        
def droite_2(landmark):
    #definition des droites
    coef = (landmark[0]-landmark[6])/(landmark[1]-landmark[7])
    ordonnee = landmark[0] - coef*landmark[1]
    return coef,ordonnee

def droite_3(landmark):
    #definition des droites
    coef = (landmark[2]-landmark[8])/(landmark[3]-landmark[9])
    ordonnee = landmark[2] - coef*landmark[3]
    return coef,ordonnee
        
def droite_4(landmark):
    #definition des droites
    coef = (landmark[7]-landmark[9])/(landmark[6]-landmark[8])
    ordonnee = landmark[7] - coef*landmark[6]
    return coef,ordonnee

mask = np.ones(shape = np.shape(data_images))
mask_x = np.cumsum(mask,axis = 2)
mask_y = np.cumsum(mask,axis = 1)

M = map(lambda  x,y,z: droite_1(x)[0]*y+droite_1(x)[1] <= z ,land,mask_x,mask_y)
img_1 = list(M)

M = map(lambda  x,y,z: droite_2(x)[0]*z+droite_2(x)[1] <= y ,land,mask_x,mask_y)
img_2 = list(M)

M = map(lambda  x,y,z: droite_3(x)[0]*z+droite_3(x)[1] >= y ,land,mask_x,mask_y)
img_3 = list(M)

M = map(lambda  x,y,z: droite_4(x)[0]*y+droite_4(x)[1] >= z ,land,mask_x,mask_y)
img_4 = list(M)

M = map(lambda v,w,x,y,z:v*w*x*y*z,data_images,img_1,img_2,img_3,img_4)
image_center = list(M)
image_center = np.asarray(image_center)
image_center[image_center == 0] = np.nan

M = map(np.nanmean,image_center)
color_center = list(M)

color_center = (color_center - np.min(color_center))/(np.max(color_center)-np.min(color_center))

#%%




