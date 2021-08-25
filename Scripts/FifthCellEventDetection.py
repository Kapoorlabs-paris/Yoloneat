#!/usr/bin/env python
# coding: utf-8

# In[2]:



import sys
import os
import glob
sys.path.append("../NEAT")
from NEATModels import NEATStatic, nets
from NEATModels.Staticconfig import static_config
from NEATUtils import helpers
from NEATUtils.helpers import load_json
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path


# In[2]:


imagedir = '/data/u934/service_imagerie/v_kapoor/oneatgolddust/Test/Raw/'
model_dir = '/data/u934/service_imagerie/v_kapoor/CurieDeepLearningModels/OneatModels/CellNetBinning1Models/'
savedir= '/data/u934/service_imagerie/v_kapoor/oneatgolddust/Test/Raw/Saved47s4seqf32/'
model_name = 'CellNetbin1d47s4seqf32'
division_categories_json = model_dir + 'StaticCategories.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'StaticCord.json'
cordconfig = load_json(division_cord_json)
model = NEATStatic(None, model_dir , model_name,catconfig, cordconfig)
Path(savedir).mkdir(exist_ok=True)
n_tiles = (1,1)
event_threshold = 1 - 1.0E-2
iou_threshold = 0.3


# In[ ]:


Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)
for imagename in X:
     model.predict(imagename, savedir, n_tiles = n_tiles, event_threshold = event_threshold, iou_threshold = iou_threshold)


# In[3]:


# In[ ]:




