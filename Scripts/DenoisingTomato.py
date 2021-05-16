#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function, unicode_literals, absolute_import, division
import sys
import os

#To run the prediction on the CPU, else comment out this line to use the GPU
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import cv2
import numpy as np
from skimage.util import invert
get_ipython().run_line_magic('matplotlib', 'inline')
from skimage import measure
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
sys.path.append('../../Terminator/')
import glob
from csbdeep.utils import Path, normalize
from tifffile import imread
from stardist.models import StarDist2D
from TerminatorUtils.helpers import normalizeFloat, OtsuThreshold2D, save_8bit_tiff_imagej_compatible,Integer_to_border
from TerminatorUtils.helpers import Prob_to_Binary, zero_pad, multiplot, CCLabels,doubleplot, remove_big_objects
from TerminatorUtils.helpers import save_tiff_imagej_compatible, SeedStarDistWatershedAll,MaxProjectDist,SeedStarDistWatershedV2
from csbdeep.models import Config, CARE
import napari
from skimage.filters import threshold_local, threshold_mean, threshold_otsu
import pandas as pd
from pandas import DataFrame, Series  # for convenience


# In[ ]:


basedir = '/data/u934/commun/@Varun-toDenoise/RNAscope_Gui_pLgr5_pThbs1_Ecad_02062020/'
savedir = basedir + 'Denoised/'
#savesegmenteddir = savedir + '_Segmented/'
Model_Dir = '/data/u934/service_imagerie/v_kapoor/CurieDeepLearningModels/'

NoiseModelName = 'BorialisS1S2FlorisMidNoiseModel'
UNETSegmentationModelName = 'DeepOzMouseUNETSegmentation'
StardistModelName = 'DrosophilaSegmentationSmartSeeds'


max_size = 5000
denoise = True
showImage = True
NoiseModel = CARE(config = None, name = NoiseModelName, basedir = Model_Dir)
#model = StarDist2D(config = None, name = StardistModelName, basedir = Model_Dir)
#UnetModel = CARE(config = None, name = UNETSegmentationModelName, basedir = Model_Dir)
Path(savedir).mkdir(exist_ok = True)
#Path(savesegmenteddir).mkdir(exist_ok = True)


# In[ ]:


Raw_path = os.path.join(basedir, '*tif')
axes = 'ZYX'
if denoise:
  filesRaw = glob.glob(Raw_path)
  for fname in filesRaw:
    
          image = imread(fname)
          Name = os.path.basename(os.path.splitext(fname)[0])
          print('Denoising Image')
          image = NoiseModel.predict(image,axes, n_tiles = (1,2,2))
             
          save_tiff_imagej_compatible((savedir  + Name+ '.tif' ) , image, axes)
          print('Denoised Image saved, to procced to next image close the Napari viewer')


                


# In[ ]:


Raw_path = os.path.join(savedir, '*tif')
axes = 'YX'
min_size = 10
axis_norm = (0,1)  
saveaxes = 'ZYX'

filesRaw = glob.glob(Raw_path)
for fname in filesRaw:
          
          #Read image        
          image = imread(fname)
  
          originalX = image.shape[1]
          originalY = image.shape[2]  
          
          Name = os.path.basename(os.path.splitext(fname)[0])
          #Declare bunch of files  
          StarsegmentationImage = np.zeros([image.shape[0], image.shape[1], image.shape[2]])
          prob = np.zeros([image.shape[0], image.shape[1], image.shape[2]])
          Binary = np.zeros([image.shape[0], image.shape[1], image.shape[2]]) 
          Binary_Watershed = np.zeros([image.shape[0], image.shape[1], image.shape[2]]) 
          Watershed = np.zeros([image.shape[0], image.shape[1], image.shape[2]])  
          Markers = np.zeros([image.shape[0], image.shape[1], image.shape[2]]) 
          Finalimage = np.zeros([image.shape[0],image.shape[1],image.shape[2]])
          Segmented= np.zeros([image.shape[0],image.shape[1],image.shape[2]])
          #Loop over Z  
          for i in range(startZ, endZ):
            x = image[i,:]
            
            
            x[endY:,:] = 0
            x[:,endX:] = 0
            #Make sure image is 2D
            

            Segmented[i,:] = UnetModel.predict(x,axes)
            thresh = threshold_otsu(Segmented[i,:])
            Finalimage[i,:] = Segmented[i,:] > thresh 
            x = normalize(x,1,99.8,axis=axis_norm)
            x = zero_pad(x, 64, 64)
            #Get stardist, label image, details, probability map, distance map
            MidImage, details = model.predict_instances(x)
            StarsegmentationImage[i,:] = MidImage[:originalX, :originalY]
            smallprob, smalldist = model.predict(x)
            grid = model.config.grid
            midprob = cv2.resize(smallprob, dsize=(smallprob.shape[1] * grid[1] , smallprob.shape[0] * grid[0] ))
            middist = cv2.resize(smalldist, dsize=(smalldist.shape[1] * grid[1] , smalldist.shape[0] * grid[0] ))
            dist = MaxProjectDist(middist)
            prob[i,:] = dist[:originalX, :originalY] * midprob[:originalX, :originalY] 
            
            
            
     
            #Seeds from Stardist, segmentation on probability map
            Watershed[i,:], Markers[i,:] = SeedStarDistWatershedV2(prob[i,:],StarsegmentationImage[i,:].astype('uint16'),Finalimage[i,:],  model.config.grid)    
            properties = measure.regionprops(Watershed[i,:].astype('uint16'), image[i,:])
            Labelindex = [prop.label for prop in properties]  
            Watershed[i,:] = remove_big_objects(Watershed[i,:].astype('uint16'), max_size)
            StarsegmentationImage[i,:] = remove_big_objects(StarsegmentationImage[i,:].astype('uint16'), max_size)
            if i%2 == 0:
              print('Zpoint', i)
              print('Total number of Tomatos at Zpoint', i, 'is', len(Labelindex))
              multiplot(Segmented[i,:],prob[i,:],Markers[i,:], 'Mask', 'Distance Map', 'Markers', plotTitle = 'Segmentation Input' )
              multiplot(image[i,:],Watershed[i,:],StarsegmentationImage[i,:].astype('uint16'), 'Original',  'SmartSeeds', 'StarDist', plotTitle = 'Super Segmentation' )
        
              
          #Save best segmentation
          save_tiff_imagej_compatible((savesegmenteddir+ 'Stardist'  + Name+ '.tif' ) , StarsegmentationImage, saveaxes)         
          save_tiff_imagej_compatible((savesegmenteddir+ 'SmartSeeds'  + Name+ '.tif' ) , Watershed, saveaxes)
          save_tiff_imagej_compatible((savesegmenteddir+ 'UNET'  + Name+ '.tif' ) , Segmented, saveaxes)
          if showImage:  
            with napari.gui_qt():
   
    
               viewer = napari.view_image(image, name='DenoisedTomato', rgb=False)  
               label_layer = viewer.add_labels(StarsegmentationImage, name='StarDist') 
               label_layer = viewer.add_labels(Watershed, name='Star') 
               #Track segmented cells        
               arboretum.run(segmentation=Watershed)       


# In[ ]:


Raw_path = os.path.join(savesegmenteddir, '*tif')

filesRaw = glob.glob(Raw_path)
for fname in filesRaw:
   image = imread(fname)
   
   core.run(segmentation=image[:,startY:endY, startX:endX])  


# In[ ]:


import urllib.request
import json

import btrack
import arboretum
import napari

import numpy as np

get_ipython().system('pwd')
get_ipython().system('cd ./')
get_ipython().system('ls')
objects = btrack.utils.import_JSON('./objects.json')
config = btrack.utils.load_config('./cell_config.json')

# track the objects
with btrack.BayesianTracker() as tracker:

    # configure the tracker using a config file, append objects and set vol
    tracker.configure(config)
    tracker.append(objects)
    tracker.volume = ((0,1200),(0,1600),(-1e5,1e5))

    # track them and (optionally) optimize
    tracker.track_interactive(step_size=100)
    tracker.optimize()

    # get the tracks as a python list
    tracks = tracker.tracks


with napari.gui_qt():
    viewer = napari.Viewer()
    arboretum.build_plugin(viewer, tracks)


# In[ ]:




