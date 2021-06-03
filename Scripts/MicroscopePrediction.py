#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
from glob import glob
sys.path.append("../NEAT")
from NEATModels import NEATPredict, nets
from NEATModels.config import dynamic_config
from NEATUtils import helpers
from NEATUtils.helpers import load_json
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# In[2]:


imagedir = '/home/sancere/Kepler/FinalONEATTraining/ONEAT_fly_test/'
model_dir =  '/home/sancere/Kepler/FinalONEATTraining/Microneatmodel/'
model_name = 'microseqnetbin2d56'
division_categories_json = model_dir + 'MicroscopeCategories.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'MicroscopeCord.json'
cordconfig = load_json(division_cord_json)
fileextension = '*tif'

model = NEATPredict(None, model_dir , model_name,catconfig, cordconfig)


# In[3]:


n_tiles = (1,1)
event_threshold = 0.9999
iou_threshold = 0.6
nb_predictions = 3


# In[ ]:


model.predict(imagedir, [], [], 0, fileextension = fileextension, nb_prediction = nb_predictions, n_tiles = n_tiles, event_threshold = event_threshold, iou_threshold = iou_threshold)


# In[ ]:




