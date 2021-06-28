#!/usr/bin/env python
# coding: utf-8

# In[1]:



import sys
import os
import glob
sys.path.append("../NEAT")
from NEATModels import NEATDynamic, nets
from NEATModels.config import dynamic_config
from NEATUtils import helpers
from NEATUtils.helpers import load_json
from stardist.models import StarDist2D
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path


# In[2]:


imagedir = '/home/sancere/Kepler/FinalONEATTraining/NEATTest/'
model_dir = '/home/sancere/Kepler/FinalONEATTraining/PreMicroneatmodel/'
savedir= '/home/sancere/Kepler/FinalONEATTraining/NEATTest/Save/'
markerdir = '/home/sancere/Kepler/FinalONEATTraining/NEATTest/Save/Markers/'
model_name = 'premicroseqnetbin2d38lstm8'
star_model_name = '/home/sancere/Kepler/FinalONEATTraining/Everydayneatmodel/bin2stardist'
division_categories_json = model_dir + 'MicroscopeCategories.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'MicroscopeCord.json'
cordconfig = load_json(division_cord_json)
model = NEATDynamic(None, model_dir , model_name,catconfig, cordconfig)
starmodel = StarDist2D(config = None, name = star_model_name, basedir = model_dir)
Path(savedir).mkdir(exist_ok=True)
n_tiles = (1,1)
event_threshold = 1.0 - 1.0E-7
iou_threshold = 0.6
yolo_v2 = False


# # In the code block below compute the markers and make a dictionary for each image

# In[ ]:


Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)

marker_dict = {}
for imagename in X:
     markers, markers_tree, density_location =  model.get_markers(imagename, starmodel,savedir, n_tiles = n_tiles, markerdir = markerdir)
     
     marker_dict[imagename] = [markers, markers_tree, density_location]
     


# # In the cell below use the computed dictionary and apply division model detection

# In[ ]:


Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)
for imagename in X:
     
     markers, markers_tree, density_location = marker_dict[imagename]
     model.predict(imagename,markers, markers_tree, density_location, savedir, n_tiles = n_tiles, event_threshold = event_threshold, iou_threshold = iou_threshold)


# In[3]:



model.showNapari(imagedir, savedir, yolo_v2)


# In[ ]:




