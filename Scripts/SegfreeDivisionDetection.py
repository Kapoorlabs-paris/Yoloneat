#!/usr/bin/env python
# coding: utf-8

# In[1]:



import sys
import os
import glob
sys.path.append("../NEAT")
from NEATModels import NEATDynamicSegFree, nets
from NEATModels.config import dynamic_config
from NEATUtils import helpers
from NEATUtils.helpers import load_json
from stardist.models import StarDist2D
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path


# In[2]:


imagedir = '/home/sancere/Kepler/FinalONEATTraining/NEATTest/'
model_dir = '/home/sancere/Kepler/FinalONEATTraining/PreMicroneatmodel/'
savedir= '/home/sancere/Kepler/FinalONEATTraining/NEATTest/Saved56l16/'
model_name = 'premicroseqnetbin2d56'
star_model_name = '/home/sancere/Kepler/FinalONEATTraining/Everydayneatmodel/bin2stardist/'
division_categories_json = model_dir + 'MicroscopeCategories.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'MicroscopeCord.json'
cordconfig = load_json(division_cord_json)
model = NEATDynamicSegFree(None, model_dir , model_name,catconfig, cordconfig)
starmodel = StarDist2D(config = None, name = star_model_name, basedir = model_dir)
Path(savedir).mkdir(exist_ok=True)
n_tiles = (1,1)
event_threshold = 1.0 - 1.0E-7
iou_threshold = 0.4
yolo_v2 = False


# # In the code block below compute the markers and make a dictionary for each image

# In[ ]:





Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)

for imagename in X:
     
     model.predict(imagename, savedir, n_tiles = n_tiles, event_threshold = event_threshold, iou_threshold = iou_threshold)


# In[3]:




# In[ ]:




