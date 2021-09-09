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
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from pathlib import Path


# In[2]:


imagedir = '/data/u934/service_imagerie/v_kapoor/oneatgolddust/Test/Bin2Test/'
model_dir = '/data/u934/service_imagerie/v_kapoor/CurieDeepLearningModels/OneatModels/CellNetBinning2Models/'
savedir= '/data/u934/service_imagerie/v_kapoor/oneatgolddust/Test/Bin2Test/Save38resf32/'
model_name = 'CellNetbin2d38resf32'
division_categories_json = model_dir + 'StaticCategories.json'
catconfig = load_json(division_categories_json)
division_cord_json = model_dir + 'StaticCord.json'
cordconfig = load_json(division_cord_json)
model = NEATStatic(None, model_dir , model_name,catconfig, cordconfig)
Path(savedir).mkdir(exist_ok=True)
n_tiles = (1,1)
event_threshold = [1,0.9,0.9,1,1,1]
iou_threshold = 0.1


# In[ ]:


Raw_path = os.path.join(imagedir, '*tif')
X = glob.glob(Raw_path)
for imagename in X:
     model.predict(imagename, savedir, event_threshold, n_tiles = n_tiles, iou_threshold = iou_threshold)


# In[3]:





# In[ ]:




