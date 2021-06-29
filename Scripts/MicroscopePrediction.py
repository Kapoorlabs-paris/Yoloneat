#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
from glob import glob
sys.path.append("../NEAT")
from NEATModels import NEATPredict, nets
from NEATModels.config import dynamic_config
from NEATUtils import helpers
from NEATUtils.helpers import load_json
from csbdeep.models import ProjectionCARE
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# In[ ]:


Z_imagedir = '/data/u934/service_imagerie/v_kapoor/FinalONEATTraining/Z_ONEAT_fly_test/'
imagedir = '/data/u934/service_imagerie/v_kapoor/FinalONEATTraining/ONEAT_fly_test/'
model_dir =  '/data/u934/service_imagerie/v_kapoor/FinalONEATTraining/Microneatmodel/'
model_name = 'microseqnetbin2d65'
projection_model_name = 'projectionmodelbin2'
division_categories_json = model_dir + 'MicroscopeCategories.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'MicroscopeCord.json'
cordconfig = load_json(division_cord_json)
fileextension = '*TIF'


model = NEATPredict(None, model_dir , model_name,catconfig, cordconfig)
projection_model = ProjectionCARE(config = None, name = projection_model_name, basedir = model_dir)



n_tiles = (1,1)
Z_n_tiles = (1,1,1)
event_threshold = 1 - 1.0E-07
iou_threshold = 0.01
nb_predictions = 10



model.predict(imagedir, [], [], Z_imagedir, [], [], 0, 0, fileextension = fileextension, nb_prediction = nb_predictions, n_tiles = n_tiles, Z_n_tiles = Z_n_tiles, event_threshold = event_threshold, iou_threshold = iou_threshold, projection_model = projection_model)

