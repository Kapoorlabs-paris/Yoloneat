#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:53:01 2020

@author: Varun Kapoor
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:25:10 2020

@author: aimachine
"""
import argparse
import numpy as np
class StaticNeatConfig(argparse.Namespace):
    
    def __init__(self, npz_directory = None, npz_name = None, npz_val_name = None, key_categories = None, key_cord = None, residual = True, gridx = 1, gridy = 1, Imagex = 128, Imagey = 128, nboxes = 1,  depth = 29, start_kernel = 3, mid_kernel = 3,startfilter = 32, 
                 epochs =100, box_vector = 5, learning_rate = 1.0E-4, batch_size = 10, model_name = 'NEATModel', yolo_v0 = True, multievent = True,  **kwargs):
        


           self.npz_directory = npz_directory
           self.npz_name = npz_name
           self.npz_val_name = npz_val_name
           self.key_categories = key_categories
           self.key_cord = key_cord
           self.residual = residual
           self.depth = depth
           self.start_kernel = start_kernel
           self.mid_kernel = mid_kernel
           self.startfilter = startfilter
           self.gridx = gridx
           self.gridy = gridy
           self.Imagex = Imagex
           self.Imagey = Imagey
           self.nboxes = nboxes
           self.epochs = epochs
           self.yolo_v0 = yolo_v0
           self.categories = len(key_categories)
           self.box_vector = box_vector
           self.learning_rate = learning_rate
           self.batch_size = batch_size
           self.model_name = model_name
           self.multievent = multievent
          
           self.is_valid()      
           
    def to_json(self):

         config = {
                 'residual' : self.residual,
                 'depth' : self.depth,
                 'start_kernel' : self.start_kernel,
                 'mid_kernel' : self.mid_kernel,
                 'startfilter' : self.startfilter,
                 'gridX' : self.gridx,
                 'gridY' : self.gridy,
                 'ImageX' : self.Imagex,
                 'ImageY' : self.Imagey,
                 'nboxes' : self.nboxes,
                 'epochs' : self.epochs,
                 'categories' : self.categories,
                 'box_vector' : self.box_vector,
                 'learning_rate' : self.learning_rate,
                 'batch_size' : self.batch_size,
                 'ModelName' : self.ModelName,
                 'multievent' : self.multievent,
                 'yolo_v0': self.yolo_v0
                 
                 }
         for (k,v) in self.key_categories.items():
             config[k] = v
             
         for (k,v) in self.key_coord.items():
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
            ok['depth']         = _is_int(self.depth,1)
            ok['start_kernel']       = _is_int(self.start_kernel,1)
            ok['mid_kernel']         = _is_int(self.mid_kernel,1)
            ok['startfilter']        = _is_int(self.startfilter, 1)
            ok['epochs']        = _is_int(self.epochs, 1)
            ok['nboxes']       = _is_int(self.nboxes, 1)
            ok['gridx'] = _is_int(self.gridx, 1)
            ok['gridy'] = _is_int(self.gridy, 1)
            ok['Imagex'] = _is_int(self.Imagex, 1)
            ok['Imagey'] = _is_int(self.Imagey, 1)
            
            ok['learning_rate'] = np.isscalar(self.learning_rate) and self.learning_rate > 0
            ok['multievent'] = isinstance(self.multievent,bool)
            ok['categories'] =  _is_int(len(self.key_categories), 1)
            ok['box_vector'] =  _is_int(self.box_vector, 1)
            if return_invalid:
                return all(ok.values()), tuple(k for (k,v) in ok.items() if not v)
            else:
                return all(ok.values())
                   

        