#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 15:25:10 2020

@author: aimachine
"""
import argparse
import numpy as np

class NeatConfig(argparse.Namespace):
    
    def __init__(self, residual = True, simple = False, catsimple = False, depth = 29, start_kernel = 7, mid_kernel = 3, startfilter = 48, lstm = 16, epochs =100, learning_rate = 1.0E-4, batch_size = 10, ModelName = 'NEATModel',  **kwargs):
        
        
           
           self.residual = residual
           self.simple = simple
           self.catsimple = catsimple
           self.depth = depth
           self.start_kernel = start_kernel
           self.mid_kernel = mid_kernel
           self.startfilter = startfilter
           self.lstm = lstm
           self.epochs = epochs
           self.learning_rate = learning_rate
           self.batch_size = batch_size
           self.ModelName = ModelName
           self.is_valid()
    

    def to_json(self):

         config = {
                 'ModelName' : self.ModelName,
                 'residual' : self.residual,
                 'simple' : self.simple,
                 'catsimple' : self.catsimple,
                 'depth' : self.depth,
                 'start_kernel' : self.start_kernel,
                 'mid_kernel' : self.mid_kernel,
                 'startfilter' : self.startfilter,
                 'lstm' : self.lstm,
                 'epochs' : self.epochs,
                 'learning_rate' : self.learning_rate,
                 'batch_size' : self.batch_size
                 }
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
            ok['simple'] = isinstance(self.simple,bool)
            ok['catsimple'] = isinstance(self.catsimple,bool)
            ok['depth']         = _is_int(self.depth,1)
            ok['start_kernel']       = _is_int(self.start_kernel,1)
            ok['mid_kernel']         = _is_int(self.mid_kernel,1)
            ok['startfilter']        = _is_int(self.startfilter, 1)
            ok['lstm']         = _is_int(self.lstm,1)
            ok['epochs']        = _is_int(self.epochs, 1)
            ok['learning_rate'] = np.isscalar(self.learning_rate) and self.learning_rate > 0
    
            
    
            if return_invalid:
                return all(ok.values()), tuple(k for (k,v) in ok.items() if not v)
            else:
                return all(ok.values())
                   

           
           
           

        
