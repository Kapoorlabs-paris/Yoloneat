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
    
def adjust_scale_prediction(y_pred, cell_grid, anchors,grid_h, grid_w, box_vector, categories, version = 1):
    
    
    nboxes = len(anchors)//2
    
    pred_nboxes = K.reshape(y_pred[...,categories:], (-1, grid_h * grid_w, nboxes, box_vector))
    
    pred_box_xy = tf.sigmoid(pred_nboxes[..., :2]) + cell_grid
    
    pred_box_wh = tf.exp(pred_nboxes[..., 2:4]) * np.reshape(anchors,[1,1,1,boxes,2])
    
    
    pred_box_conf = tf.sigmoid(y_pred[..., 4])
     
        
    pred_box_class = y_pred[..., :categories]
    
    return pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class

def extract_ground_truth(y_true, nboxes, grid_h, grid_w, box_vector, categories, version = 1):

    true_nboxes = K.reshape(y_true[...,categories:], (-1, grid_h * grid_w, nboxes, box_vector))
    
    true_box_xy    = true_nboxes[..., 0:2]  
    true_box_wh    = true_nboxes[..., 2:4] 
    if version > 1:
         true_box_conf  = y_true[...,4]
    else:
        true_box_conf = 1.0
    true_box_class = y_true[..., :categories]

    return true_box_xy, true_box_wh, true_box_conf, true_box_class
    
def compute_iou(true_xy, true_wh, pred_xy, pred_wh):

       
    intersect_wh = K.maximum(K.zeros_like(pred_wh), (pred_wh + true_wh)/2 - K.square(pred_xy - true_xy))
    intersect_area = intersect_wh[...,0] * intersect_wh[...,1]
    true_area = true_wh[...,0] * true_wh[...,1]
    pred_area = pred_wh[...,0] * pred_wh[...,1]
    union_area = pred_area + true_area - intersect_area
    iou_scores = tf.truediv(intersect_area , union_area)

    
    return iou_scores

def compute_boxconf_iou(true_box_conf, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):

    iou_scores        =  compute_iou(true_box_xy,true_box_wh,
                                            pred_box_xy,pred_box_wh)
    true_box_conf_IOU = iou_scores * true_box_conf
    
    return true_box_conf_IOU

def compute_best_ious(pred_box_xy, pred_box_wh, true_xy, true_wh):

    
    
    iou_scores  =  compute_iou(true_xy,
                                      true_wh,
                                      pred_box_xy,
                                      pred_box_wh)   

    best_ious = K.max(iou_scores, axis= -1)
    
    return best_ious

def compute_conf_loss(best_ious, true_box_conf, true_box_conf_iou,pred_box_conf):
    
    indicator_noobj = K.cast(best_ious < 0.6, np.float32) * (1 - true_box_conf) * lambdanoobject
    indicator_obj = true_box_conf * lambdaobject
    indicator_o = indicator_obj + indicator_noobj

    loss_obj = K.sum(K.square(true_box_conf_iou-pred_box_conf) * indicator_o)#, axis=[1,2,3])

    return loss_obj 

def compute_conf_mask(best_ious, true_box_conf, true_box_conf_iou, lambdanoobject, lambdaobject):
    
    
    conf_mask = tf.to_float(best_ious < 0.6) * (1 - true_box_conf) * lambdanoobject
    conf_mask = conf_mask + true_box_conf_iou * lambdanoobject

    return conf_mask

def calc_loss_xywh(true_box_conf, true_box_xy, pred_box_xy, true_box_wh, pred_box_wh):

    
    coord_mask  = tf.expand_dims(true_box_conf, axis=-1) * lambdacoord 
    nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
    loss_xy      = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_wh      = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
    loss_xywh = loss_xy + loss_wh
    return loss_xywh, coord_mask

def calc_loss_class(true_box_conf, true_box_class, pred_box_class, entropy):

    
    class_mask   = true_box_conf  * lambdaclass
    
    nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
    
    
    if entropy == 'binary':
        loss_class = K.mean(K.binary_crossentropy(true_box_class, pred_box_class), axis=-1)
    if entropy == 'notbinary':
         
        loss_class   = K.mean(K.categorical_crossentropy(true_box_class, pred_box_class), axis=-1)
    loss_class   = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)

    return loss_class


def yolo_loss_v0(categories, grid_h, grid_w, nboxes, box_vector, entropy):
    
    def loss(y_true, y_pred):    

            
        true_box_class = y_true[...,0:categories]
        pred_box_class = y_pred[...,0:categories]
        
        
        pred_box_xy = y_pred[...,categories:categories + 2] 
        
        true_box_xy = y_true[...,categories:categories + 2] 
        
        
        pred_box_wh = y_pred[...,categories + 2:] 
        
        true_box_wh = y_true[...,categories + 2:] 
        


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




def static_yolo_loss(categories, grid_h, grid_w, anchors, box_vector, entropy):
    
    def loss(y_true, y_pred):    

            nboxes = int(len(anchors)/2)
            # Get the cell grid
            cell_grid = get_cell_grid(grid_h, grid_w, nboxes)  
            
            #Extract the ground truth 
            true_box_xy, true_box_wh, true_box_conf, true_box_class = extract_ground_truth(y_true, nboxes, grid_h, grid_w, box_vector, categories)
            
            #Scale the prediction variables
            pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = adjust_scale_prediction(y_pred, cell_grid, anchors,grid_h, grid_w, box_vector, categories)
        
            # xyhw loss
            loss_xywh, coord_mask = calc_loss_xywh(true_box_conf, true_box_xy, pred_box_xy, true_box_wh, pred_box_wh)
            
            # class loss
            loss_class = calc_loss_class(true_box_conf, true_box_class, pred_box_class, entropy)
            
            # compute iou
            true_box_conf_iou = compute_iou(true_box_xy, true_box_wh, pred_box_xy, pred_box_wh)
            
            # best ious
            best_ious = compute_best_ious(pred_box_xy, pred_box_wh, true_box_xy, true_box_wh)
            
            #conf loss
            loss_conf = compute_conf_loss(best_ious, true_box_conf, true_box_conf_iou, pred_box_conf)
            
           

            # Adding it all up   
            combinedloss = (loss_xywh + loss_conf + loss_class)
        

     
            return combinedloss 
        
    return loss    