"""
B. FREICHE 2024-04-23
File for the visualisation of the latent spaces learned with Hierarchical GP-LVM on CelebA

1) GP-LVM 10 dims on landmarks
2) GP-LVM 10 dims on images
3) Hierarchical GP-LVM 10 dims on landmarks -> images
"""

import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

# File with the computation of the colomaps+some display function to show latent space with the images
exec(open('C:/Users/b_freich/Documents/Research/Code/Code_these_pour_ND/gpHierarchy/experiments/colormap_experience_hierarchie.py').read())

#%% 1

with open('C:/Users/b_freich/Documents/Research/Code/Code_these_pour_ND/Results/LANDMARKS/dim_3/spaces.pkl','rb') as f:
    out = pkl.load(f)
    
X = out['X'][0]

# Displays the latent space of LANDMARKS colored by orientation, smile and center intensity 
plt.figure(figsize = (20,5))
plt.subplot(131)
plt.scatter(X[:,0],X[:,1],c= color_x,cmap = 'Spectral_r',s = 100)
plt.title('Orientation')
plt.colorbar()

plt.subplot(132)
plt.scatter(X[:,0],X[:,2],c= color_smile,cmap = 'rainbow',s = 100)
plt.title('Smile')
plt.colorbar()

plt.subplot(133)
plt.scatter(X[:,0],X[:,1],c= color_center,cmap = 'jet',s = 100)
plt.title('Center intensity')
plt.colorbar()
plt.show()

#%% Displays the landmarks corresponding to the 2 first dims (x,y) of the latent space of LANDMARKS
paths = [land[i] for i in range(len(land))]
y = X[:,0]
x = X[:,1]

fig, ax = plt.subplots()

ax.scatter(x, y, color = 'white')
fig.set_size_inches((15,15))
for x0, y0, path in zip(x, y,paths):
    ab = AnnotationBbox(getImage(getScatterImage(path,size = 5),zoom = 0.4,cmap =  mpl.cm.gray_r,norm=mpl.colors.Normalize(vmin=0, vmax=1)), (x0, y0),frameon = False)
    ax.add_artist(ab)
    # for an other display: 
    # size is the pixel size of the dots corresponding to the landmarks
    # zoom is the size of the display of the image
    # cmap is the colormap; Normalize allow to change vmin and vmax for cmap
    
ax = plt.gca()
ax_lims = (min(np.min(ax.get_ylim()),np.min(ax.get_xlim())),max(np.max(ax.get_ylim()),np.max(ax.get_xlim())))
ax.set_xlim(ax_lims)
ax.set_ylim(ax_lims)
plt.show()
#%% 2
with open('C:/Users/b_freich/Documents/Research/Code/Code_these_pour_ND/Results/IMAGES/dim_10/spaces.pkl','rb') as f:
    out = pkl.load(f)
    
X = out['X'][0]

plt.figure(figsize = (20,5))
plt.subplot(131)
plt.scatter(X[:,0],X[:,1],c= color_x,cmap = 'Spectral_r',s = 100)
plt.title('Orientation')
plt.colorbar()

plt.subplot(132)
plt.scatter(X[:,0],X[:,2],c= color_smile,cmap = 'rainbow',s = 100)
plt.title('Smile')
plt.colorbar()

plt.subplot(133)
plt.scatter(X[:,0],X[:,1],c= color_center,cmap = 'jet',s = 100)
plt.title('Center intensity')
plt.colorbar()
plt.show()

#%% Displays the images corresponding to the 2 first dims (x,y) of the latent space of IMAGES
paths = [data_images[i] for i in range(len(data_images))]
y = X[:,0]
x = X[:,1]

fig, ax = plt.subplots()

ax.scatter(x, y, color = 'white')
fig.set_size_inches((15,15))
for x0, y0, path in zip(x, y,paths):
    ab = AnnotationBbox(getImage(path,zoom = 0.4), (x0, y0),frameon = False)
    ax.add_artist(ab)
ax = plt.gca()
ax_lims = (min(np.min(ax.get_ylim()),np.min(ax.get_xlim())),max(np.max(ax.get_ylim()),np.max(ax.get_xlim())))
ax.set_xlim(ax_lims)
ax.set_ylim(ax_lims)
plt.show()
#%% 3
with open('C:/Users/b_freich/Documents/Research/Code/Code_these_pour_ND/Results/HIERARCHY/dim_10/link_1/spaces.pkl','rb') as f:
    out = pkl.load(f)
    
X = out['X'][0]

plt.figure(figsize = (20,5))
plt.subplot(131)
plt.scatter(X[:,0],X[:,1],c= color_x,cmap = 'Spectral_r',s = 100)
plt.title('Orientation')
plt.colorbar()

plt.subplot(132)
plt.scatter(X[:,5],X[:,7],c= color_smile,cmap = 'rainbow',s = 100)
plt.title('Smile')
plt.colorbar()

plt.subplot(133)
plt.scatter(X[:,0],X[:,1],c= color_center,cmap = 'jet',s = 100)
plt.title('Center intensity')
plt.colorbar()
plt.show()

#%% Displays the images corresponding to the 2 first dims (x,y) of the latent space of HIERARCHY
paths = [data_images[i] for i in range(len(data_images))]
y = X[:,0]
x = X[:,1]

fig, ax = plt.subplots()

ax.scatter(x, y, color = 'white')
fig.set_size_inches((15,15))
for x0, y0, path in zip(x, y,paths):
    ab = AnnotationBbox(getImage(path,zoom = 0.4), (x0, y0),frameon = False)
    ax.add_artist(ab)
ax = plt.gca()
ax_lims = (min(np.min(ax.get_ylim()),np.min(ax.get_xlim())),max(np.max(ax.get_ylim()),np.max(ax.get_xlim())))
ax.set_xlim(ax_lims)
ax.set_ylim(ax_lims)
plt.show()