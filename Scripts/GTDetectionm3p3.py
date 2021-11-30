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

from csbdeep.models import Config, CARE

os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path


# In[2]:


imagedir =  '/data/u934/service_imagerie/v_kapoor/WildTypeTest/wt12/'
model_dir = '/data/u934/service_imagerie/v_kapoor/CurieDeepLearningModels/WinnerOneatModels/'
savedir= '/data/u934/service_imagerie/v_kapoor/WildTypeTest/wt12/GTmode_testm3p3_ht_low/'
markerdir = '/data/u934/service_imagerie/v_kapoor/WildTypeTest/wt12/Markers/'
model_name = 'Cellsplitdetectorm3p3'

division_categories_json = model_dir + 'Cellsplitcategories.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'Cellsplitcord.json'
cordconfig = load_json(division_cord_json)
model = NEATDynamic(None, model_dir , model_name,catconfig, cordconfig)
Path(savedir).mkdir(exist_ok=True)
n_tiles = (4,4)
event_threshold = 1-1.0E-04 #[1,0.99999,0.99999,1,1,1]
iou_threshold = 0.9
yolo_v2 = False
downsample = 2
remove_markers = False
# # In the code block below compute the markers and make a dictionary for each image

# In[ ]:


Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)
X = sorted(X)
marker_dict = {}
for imagename in X:
   
     markers, marker_tree, density = model.get_markers(imagename, None, savedir, n_tiles = n_tiles, markerdir=markerdir, star=True, downsample = downsample) 
     model.predict(imagename,markers, marker_tree, density,  savedir, n_tiles = n_tiles, event_threshold = event_threshold, iou_threshold = iou_threshold, remove_markers = remove_markers, downsample = downsample)


# In[3]:




# In[ ]:



