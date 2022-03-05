#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 28 13:49:35 2021

@author: vkapoor
"""

import numpy as np
from ..NEATUtils.helpers import yoloprediction, normalizeFloatZeroOne, DownsampleData, save_dynamic_csv, dynamic_nms
import os
import math
import tensorflow as tf
from tqdm import tqdm
from ..NEATModels.nets import Concat
from keras.models import load_model
from tifffile import imread, imwrite
from scipy import ndimage
from scipy import spatial
import matplotlib.pyplot  as plt
import h5py
from neat_goldstandard import NEATDynamic

class NEATSynamic(NEATDynamic):
    

    """
    Parameters
    ----------
    
    NpzDirectory : Specify the location of npz file containing the training data with movies and labels
    
    TrainModelName : Specify the name of the npz file containing training data and labels
    
    ValidationModelName :  Specify the name of the npz file containing validation data and labels
    
    categories : Number of action classes
    
    Categories_Name : List of class names and labels
    
    model_dir : Directory location where trained model weights are to be read or written from
    
    model_name : The h5 file of CNN + LSTM + Dense Neural Network to be used for training
    
    model_keras : The model as it appears as a Keras function
    
    model_weights : If re-training model_weights = model_dir + model_name else None as default
    
    lstm_hidden_units : Number of hidden uniots for LSTm layer, 64 by default
    
    epochs :  Number of training epochs, 55 by default
    
    batch_size : batch_size to be used for training, 20 by default
    
    
    
    """
    
    
    def __init__(self, config, model_dir, model_name, catconfig = None, cordconfig = None):

        super().__init__(config = config, model_dir = model_dir, model_name = model_name, catconfig = catconfig, cordconfig = cordconfig)
        
        

        
    def predict(self,imagename, savedir, n_tiles = (1,1), overlap_percent = 0.8, event_threshold = 0.5, iou_threshold = 0.1, thresh = 5, downsamplefactor = 1, maskimagename = None, maskfilter = 10):
        

        self.predict(imagename,savedir,n_tiles = n_tiles, overlap_percent = overlap_percent, event_threshold = event_threshold, iou_threshold = iou_threshold, 
        thresh = thresh, downsamplefactor = downsamplefactor, maskimagename = maskimagename, maskfilter = maskfilter, density_veto = None, markers = None, marker_tree = None,
        density_location = None, remove_markers = None )
        
                                
                            
    def remove_marker_locations(self, tcenter, location):

                     tree, indices = self.marker_tree[str(int(tcenter))]
                     try:
                        indices.remove(location)
                     except:
                         pass
                     tree = spatial.cKDTree(indices)
                    
                     #Update the tree
                     self.marker_tree[str(int(tcenter))] =  [tree, indices]
            
                              
   
    

        