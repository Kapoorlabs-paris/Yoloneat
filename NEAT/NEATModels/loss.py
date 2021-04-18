#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:05:34 2021
@author: vkapoor
"""

import tensorflow as tf
import numpy as np
import os, sys
from keras import backend as K

lambdaobject = 1
lambdanoobject = 1
lambdacoord = 5
lambdaclass = 1
grid_h = 1
grid_w = 1

def get_cell_grid(grid_h, grid_w, boxes):
    
    cell_grid = np.array([ [[float(x),float(y)]]*boxes   for y in range(grid_h) for x in range(grid_w)])
    
    return cell_grid
    
def extract_ground_pred(y_pred, categories):

        pred_box_class = y_pred[...,0:categories]
        
        pred_box_xy = y_pred[...,categories:categories + 2] 
        
        pred_box_wh = y_pred[...,categories + 2:categories + 4] 
        
        pred_box_conf = y_pred[...,categories + 4]
        
        return pred_box_class, pred_box_xy, pred_box_wh, pred_box_conf

def extract_ground_truth(y_true, categories):

        true_box_class = y_true[...,0:categories]
        
        true_box_xy = y_true[...,categories:categories + 2] 
        
        true_box_wh = y_true[...,categories + 2:categories + 4] 
        
        true_box_conf = y_true[...,categories + 4] 
        
        
        return true_box_class, true_box_xy, true_box_wh, true_box_conf


def compute_conf_loss(pred_box_wh, true_box_wh, pred_box_xy,true_box_xy,true_box_conf,pred_box_conf):
    
# compute the intersection of all boxes at once (the IOU)
        intersect_wh = K.maximum(K.zeros_like(pred_box_wh), (pred_box_wh + true_box_wh)/2 - K.square(pred_box_xy - true_box_xy) )
        intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
        true_area = true_box_wh[...,0] * true_box_wh[...,1]
        pred_area = pred_box_wh[...,0] * pred_box_wh[...,1]
        union_area = pred_area + true_area - intersect_area
        iou = tf.truediv(intersect_area , union_area)
        best_ious = K.max(iou, axis= -1)
        loss_conf = K.sum(K.square(true_box_conf*best_ious - pred_box_conf), axis=-1)

        loss_conf = loss_conf * lambdaobject

        return loss_conf 

def calc_loss_xywh(true_box_conf, true_box_xy, pred_box_xy, true_box_wh, pred_box_wh, y_true_conf):

    
    loss_xy      = K.sum(K.sum(K.square(true_box_xy - pred_box_xy), axis = -1), axis = -1)
    loss_wh      = K.sum(K.sum(K.square(K.sqrt(true_box_wh) - K.sqrt(pred_box_wh)), axis=-1), axis=-1)
    loss_xywh = (loss_xy + loss_wh)
    loss_xywh = lambdacoord * loss_xywh
    return loss_xywh

def calc_loss_class(true_box_class, pred_box_class, entropy):

    
        if entropy == 'binary':
            loss_class = K.mean(K.binary_crossentropy(true_box_class, pred_box_class), axis=-1)
        if entropy == 'notbinary':
            loss_class   = K.mean(K.categorical_crossentropy(true_box_class, pred_box_class), axis=-1)

        loss_class   = loss_class * lambdaclass 

        return loss_class


def yolo_loss_v0(categories, grid_h, grid_w, nboxes, box_vector, entropy):
    
    def loss(y_true, y_pred):    

            
        true_box_class = y_true[...,0:categories]
        pred_box_class = y_pred[...,0:categories]
        
        
        pred_box_xy = y_pred[...,categories:categories + 2] 
        
        true_box_xy = y_true[...,categories:categories + 2] 
        
        
        pred_box_wh = y_pred[...,categories + 2:categories + 4] 
        
        true_box_wh = y_true[...,categories + 2:categories + 4] 
        


        loss_xy      = K.sum(K.sum(K.square(true_box_xy - pred_box_xy), axis = -1), axis = -1)
        loss_wh      = K.sum(K.sum(K.square(K.sqrt(true_box_wh) - K.sqrt(pred_box_wh)), axis=-1), axis=-1)
        loss_xywh = (loss_xy + loss_wh)
        loss_xywh = lambdacoord * loss_xywh

        if entropy == 'binary':
            loss_class = K.mean(K.binary_crossentropy(true_box_class, pred_box_class), axis=-1)
        if entropy == 'notbinary':
            loss_class   = K.mean(K.categorical_crossentropy(true_box_class, pred_box_class), axis=-1)

        loss_class   = loss_class * lambdaclass 

        combinedloss = loss_xywh + loss_class
            
        return combinedloss 
        
    return loss 




def static_yolo_loss(categories, grid_h, grid_w, nboxes, box_vector, entropy):
    
    def loss(y_true, y_pred):    

        true_box_class, true_box_xy, true_box_wh, true_box_conf = extract_ground_truth(y_true, categories)
        pred_box_class, pred_box_xy, pred_box_wh, pred_box_conf = extract_ground_pred(y_pred, categories)

        loss_xywh = calc_loss_xywh(true_box_conf, true_box_xy, pred_box_xy, true_box_wh, pred_box_wh, true_box_conf)

        loss_class   = calc_loss_class(true_box_class, pred_box_class, entropy)

        loss_conf = compute_conf_loss(pred_box_wh, true_box_wh, pred_box_xy,true_box_xy,true_box_conf,pred_box_conf)
        # Adding it all up   
        combinedloss = (loss_xywh + loss_conf + loss_class)



        return combinedloss 
        
    return loss    