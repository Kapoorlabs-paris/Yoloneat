#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 10:15:14 2021

@author: aimachine
"""
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
from glob import glob
sys.path.append("../NEAT")
from NEATModels import NEATStaticPredict, nets
from NEATModels.config import dynamic_config
from NEATUtils import helpers
from NEATUtils.helpers import load_json
from csbdeep.models import ProjectionCARE
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# In[2]:


Z_imagedir = '/data/u934/service_imagerie/v_kapoor/FinalONEATTraining/Z_ONEAT_fly_test/'
imagedir = '/data/u934/service_imagerie/v_kapoor/FinalONEATTraining/ONEAT_fly_test/'
model_dir =  '/data/u934/service_imagerie/v_kapoor/CurieDeepLearningModels/OneatModels/CellNetBinning2Models/'
model_name = 'CellNetbin2d38resf32'
projection_model_name = 'projectionmodelbin2'
division_categories_json = model_dir + 'StaticCategories.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'StaticCord.json'
cordconfig = load_json(division_cord_json)
fileextension = '*TIF'

model = NEATStaticPredict(None, model_dir , model_name,catconfig, cordconfig)
projection_model = None #ProjectionCARE(config = None, name = projection_model_name, basedir = model_dir)


# In[3]:


n_tiles = (1,1)
Z_n_tiles = (1,2,2)
event_threshold = 0.9999
iou_threshold = 0.1
nb_predictions = 30


# In[4]:


model.predict(imagedir, {}, {}, Z_imagedir, [], [],  0, 0, fileextension = fileextension, nb_prediction = nb_predictions, n_tiles = n_tiles, Z_n_tiles = Z_n_tiles, event_threshold = event_threshold, iou_threshold = iou_threshold, projection_model = projection_model)


# In[ ]:




