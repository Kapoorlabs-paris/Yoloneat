#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
from glob import glob
sys.path.append("../NEAT")
from NEATModels import NEATFocusPredict, nets
from NEATModels.config import dynamic_config
from NEATUtils import helpers
from NEATUtils.helpers import load_json

os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


# In[2]:



Z_imagedir = '/data/u934/service_imagerie/v_kapoor/FinalONEATTraining/Z_ONEAT_fly_test/'
imagedir = '/data/u934/service_imagerie/v_kapoor/FinalONEATTraining/ONEAT_fly_test/'
model_dir =  '/data/u934/service_imagerie/v_kapoor/CurieDeepLearningModels/OneatModels/Focusoneatmodels/'
model_name = 'cadhistoned29s5f16res'

focus_categories_json = model_dir + 'FocusCategories.json'
catconfig = load_json(focus_categories_json)
focus_cord_json = model_dir + 'FocusCord.json'
cordconfig = load_json(focus_cord_json)
fileextension = '*TIF'

model = NEATFocusPredict(None, model_dir , model_name,catconfig, cordconfig)


# In[3]:


Z_n_tiles = (1,2,2)

nb_predictions = 5


# In[4]:


model.predict(imagedir, Z_imagedir, [], [],  0, 0, fileextension = fileextension, nb_prediction = nb_predictions,  Z_n_tiles = Z_n_tiles)


# In[ ]:




