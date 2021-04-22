#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:25:10 2020

@author: aimachine
"""
import argparse
import numpy as np

class NeatConfig(argparse.Namespace):
    
    def __init__(self, npz_directory = None, npz_name = None, npz_val_name = None, key_categories = None, key_cord = None,  residual = True, imagex = 128, imagey = 128, imaget = 9, nboxes = 1,
                 depth = 29, start_kernel = 7, mid_kernel = 3, startfilter = 48, lstm = 16, epochs =100, 
                 learning_rate = 1.0E-4, batch_size = 10, model_name = 'NEATModel', yolo_v0 = True, yolo_v1 = False, yolo_v2 = False, multievent = True,  **kwargs):
        
        
           self.npz_directory = npz_directory
           self.npz_name = npz_name
           self.npz_val_name = npz_val_name
           self.key_categories = key_categories
           self.key_cord = key_cord
           self.residual = residual
           self.yolo_v0 = yolo_v0
           self.yolo_v1 = yolo_v1
           self.yolo_v2 = yolo_v2
           self.categories = len(self.key_categories)
           self.box_vector = len(self.key_cord)
           self.imagex = imagex
           self.imagey = imagey
           self.imaget = imaget
           self.depth = depth
           self.start_kernel = start_kernel
           self.mid_kernel = mid_kernel
           self.startfilter = startfilter
           self.lstm = lstm
           self.epochs = epochs
           self.learning_rate = learning_rate
           self.batch_size = batch_size
           self.model_name = model_name
           self.is_valid()
    

    def to_json(self):

         config = {
                 'model_name' : self.model_name,
                 'residual' : self.residual,
                 'multievent' : self.multievent,
                 'yolo_v0': self.yolo_v0,
                 'yolo_v1': self.yolo_v1,
                 'yolo_v2': self.yolo_v2,
                 'imagex' : self.imagex,
                 'imagey' : self.imagey,
                 'imaget' : self.imaget,
                 'nboxes' : self.nboxes,
                 'depth' : self.depth,
                 'start_kernel' : self.start_kernel,
                 'mid_kernel' : self.mid_kernel,
                 'startfilter' : self.startfilter,
                 'lstm' : self.lstm,
                 'epochs' : self.epochs,
                 'learning_rate' : self.learning_rate,
                 'batch_size' : self.batch_size
                 }
         
         for (k,v) in self.key_categories.items():
             config[k] = v
             
         for (k,v) in self.key_cord.items():
             config[k] = v
             
             
             
         return config
         
         
        
          
    def is_valid(self, return_invalid=False):
            """Check if configuration is valid.
            Returns
            -------
            bool
            Flag that indicates whether the current configuration values are valid.
            """
            def _is_int(v,low=None,high=None):
              return (
                isinstance(v,int) and
                (True if low is None else low <= v) and
                (True if high is None else v <= high)
              )

            ok = {}
            ok['residual'] = isinstance(self.residual,bool)
            ok['yolo_v0'] = isinstance(self.yolo_v0,bool)
            ok['yolo_v1'] = isinstance(self.yolo_v1,bool)
            ok['yolo_v2'] = isinstance(self.yolo_v2,bool)
            ok['depth']         = _is_int(self.depth,1)
            ok['start_kernel']       = _is_int(self.start_kernel,1)
            ok['mid_kernel']         = _is_int(self.mid_kernel,1)
            ok['startfilter']        = _is_int(self.startfilter, 1)
            ok['lstm']         = _is_int(self.lstm,1)
            ok['epochs']        = _is_int(self.epochs, 1)
            ok['nboxes']       = _is_int(self.nboxes, 1)
            
            ok['imagex'] = _is_int(self.imagex, 1)
            ok['imagey'] = _is_int(self.imagey, 1)
            ok['imaget'] = _is_int(self.imaget, 1)
            
            ok['learning_rate'] = np.isscalar(self.learning_rate) and self.learning_rate > 0
            ok['multievent'] = isinstance(self.multievent,bool)
            ok['categories'] =  _is_int(len(self.key_categories), 1)
            ok['box_vector'] =  _is_int(self.box_vector, 1)
            
    
            if return_invalid:
                return all(ok.values()), tuple(k for (k,v) in ok.items() if not v)
            else:
                return all(ok.values())
                   

           
           
           

        
