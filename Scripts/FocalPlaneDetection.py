#!/usr/bin/env python
# coding: utf-8

# In[1]:



import sys
import os
import glob
sys.path.append("../NEAT")
from NEATModels import NEATFocus, nets
from NEATModels.config import dynamic_config
from NEATUtils import helpers
from NEATUtils.helpers import load_json
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path


# In[2]:


imagedir = '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/foconeatimages/'
model_dir = '/data/u934/service_imagerie/v_kapoor/CurieDeepLearningModels/OneatModels/Focusoneatmodels/'
savedir= '/data/u934/service_imagerie/v_kapoor/CurieTrainingDatasets/foconeatimages/Predictionsd29s5f16res/'

model_name = 'cadhistoned29s5f16res'
focus_categories_json = model_dir + 'FocusCategories.json'
catconfig = load_json(focus_categories_json)
focus_cord_json = model_dir + 'FocusCord.json'
cordconfig = load_json(focus_cord_json)
model = NEATFocus(None, model_dir , model_name,catconfig, cordconfig)
Path(savedir).mkdir(exist_ok=True)
n_tiles = (1,1)
interest_event = ("BestCad", "BestNuclei")


# # In the code block below compute the markers and make a dictionary for each image

# In[ ]:


Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)


for imagename in X:
     
         model.predict(imagename, savedir, interest_event, n_tiles = n_tiles)


# In[ ]:




