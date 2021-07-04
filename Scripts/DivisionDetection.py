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
from csbdeep.models import Config, CARE

os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path


# In[2]:


imagedir = '/data/u934/service_imagerie/v_kapoor/FinalONEATTraining/NEATTest/'
model_dir = '/data/u934/service_imagerie/v_kapoor/FinalONEATTraining/EverydayneatmodelV1/'
savedir= '/data/u934/service_imagerie/v_kapoor/FinalONEATTraining/NEATTest/MarkerNineFramePrediction/'
markerdir = '/data/u934/service_imagerie/v_kapoor/FinalONEATTraining/NEATTest/MarkerNineFramePrediction/Markers/'
model_name = 'divisionm4d65'
unet_model_name = '/data/u934/service_imagerie/v_kapoor/FinalONEATTraining/EverydayneatmodelV1/UNETDrosophilaMembrane/'
division_categories_json = model_dir + 'DivisionCategories.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'DivisionCord.json'
cordconfig = load_json(division_cord_json)
model = NEATDynamic(None, model_dir , model_name,catconfig, cordconfig)
unetmodel = CARE(config = None, name = unet_model_name, basedir = model_dir)
Path(savedir).mkdir(exist_ok=True)
n_tiles = (1,1)
event_threshold = 0.99
iou_threshold = 0.01
yolo_v2 = False


# # In the code block below compute the markers and make a dictionary for each image

# In[ ]:


Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)

marker_dict = {}
for imagename in X:
     markers, markers_tree, density_location =  model.get_markers(imagename, starmodel,savedir, n_tiles = n_tiles, markerdir = markerdir)
     
     model.predict(imagename,markers, markers_tree, density_location, savedir, n_tiles = n_tiles, event_threshold = event_threshold, iou_threshold = iou_threshold)


# In[3]:




# In[ ]:




