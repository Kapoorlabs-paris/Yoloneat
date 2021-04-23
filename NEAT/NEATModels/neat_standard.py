from NEATUtils import plotters
import numpy as np
from NEATUtils import helpers
from NEATUtils.helpers import save_json, load_json, yoloprediction, normalizeFloatZeroOne
from keras import callbacks
import os
from NEATModels import nets
from keras import backend as K
#from IPython.display import clear_output
from keras import optimizers
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from keras.models import load_model
from tifffile import imread, imwrite
import csv


class NEATDetection(object):
    

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
    
    
    def __init__(self, config, model_dir, model_name):

        
        self.config = config
        if self.config !=None:
                self.npz_directory = config.npz_directory
                self.npz_name = config.npz_name
                self.npz_val_name = config.npz_val_name
                self.key_catagories = config.key_catagories
                self.box_vector = config.box_vector
                self.show = config.show
                self.key_cord = config.key_cord
                self.categories = len(config.key_catagories)
                self.depth = config.depth
                self.start_kernel = config.start_kernel
                self.mid_kernel = config.mid_kernel
                self.learning_rate = config.learning_rate
                self.epochs = config.epochs
                self.residual = config.residual
                self.startfilter = config.startfilter
                self.batch_size = config.batch_size
                self.multievent = config.multievent
                self.imagex = config.imagex
                self.imagey = config.imagey
                self.nboxes = config.nboxes
                self.gridx = config.gridx
                self.gridy = config.gridy
                self.yolo_v0 = config.yolo_v0
                self.stride = config.stride
                self.lstm_hidden_unit = config.lstm
        if self.config == None:
               
                try:
                   self.config = load_json(self.model_dir + os.path.splitext(self.model_name)[0] + '_Parameter.json')
                except:
                   self.config = load_json(self.model_dir + self.model_name + '_Parameter.json')  
                   
                self.npz_directory = config['npz_directory']
                self.npz_name = config['npz_name']
                self.npz_val_name = config['npz_val_name']
                self.key_catagories = config['key_catagories']
                self.box_vector = config['box_vector']
                self.show = config['show']
                self.KeyCord = config['key_cord']
                self.categories = len(config['key_catagories'])
                self.depth = config['depth']
                self.start_kernel = config['start_kernel']
                self.mid_kernel = config['mid_kernel']
                self.learning_rate = config['learning_rate']
                self.epochs = config['epochs']
                self.residual = config['residual']
                self.startfilter = config['startfilter']
                self.batch_size = config['batch_size']
                self.multievent = config['multievent']
                self.imagex = config['imagex']
                self.imagey = config['imagey']
                self.nboxes = config['nboxes']
                self.gridx = config['gridx']
                self.gridy = config['gridy']
                self.yolo_v0 = config['yolo_v0']
                self.stride = config['stride']         
                
        self.model_dir = model_dir
        self.model_name = model_name 
        self.X = None
        self.Y = None
        self.axes = None
        self.X_val = None
        self.Y_val = None
        self.Trainingmodel = None
        self.Xoriginal = None
        self.Xoriginal_val = None
        print(self.startfilter)
    def loadData(self):
        
        (X,Y),  axes = helpers.load_full_training_data(self.NpzDirectory, self.TrainModelName, verbose= True)

        (X_val,Y_val), axes = helpers.load_full_training_data(self.NpzDirectory, self.ValidationModelName,  verbose= True)
        
        
        self.Xoriginal = X
        self.Xoriginal_val = X_val
        

                     

        self.X = X
        self.Y = Y[:,:,0]
        self.X_val = X_val
        self.Y_val = Y_val[:,:,0]
        self.axes = axes
        self.Y = self.Y.reshape( (self.Y.shape[0],1,1,self.Y.shape[1]))
        self.Y_val = self.Y_val.reshape( (self.Y_val.shape[0],1,1,self.Y_val.shape[1]))
          

              
    def TrainModel(self):
        
        input_shape = (self.X.shape[1], self.X.shape[2], self.X.shape[3], self.X.shape[4])
        
        
        Path(self.model_dir).mkdir(exist_ok=True)
        
       
        Y_rest = self.Y[:,:,:,self.categories:]
        Y_main = self.Y[:,:,:,0:self.categories-1]
 
        y_integers = np.argmax(Y_main, axis = -1)
        y_integers = y_integers[:,0,0]

        
        
        d_class_weights = compute_class_weight('balanced', np.unique(y_integers), y_integers)
        d_class_weights = d_class_weights.reshape(1,d_class_weights.shape[0])
        
        if self.residual == True and self.simple == False:
            model_keras = nets.ORNET
        if self.residual == False and self.simple == False: 
            model_keras = nets.OSNET
        if self.residual == True and self.simple == True:
            model_keras = nets.SimpleORNET
        if self.residual == False and self.simple == True:
            model_keras = nets.SimpleOSNET
        if self.residual == False and self.catsimple == True:
            model_keras = nets.CatSimpleOSNET
        if self.residual == True and self.catsimple == True:
            model_keras = nets.CatSimpleORNET
            
         
        self.Trainingmodel = model_keras(input_shape, self.categories,  unit = self.lstm_hidden_unit , box_vector = Y_rest.shape[-1] , depth = self.depth, start_kernel = self.start_kernel, mid_kernel = self.mid_kernel, startfilter = self.startfilter,  input_weights  =  self.model_weights)
        
            
        sgd = optimizers.SGD(lr=self.learning_rate, momentum = 0.99, decay=1e-6, nesterov = True)
        if self.simple == False:
          self.Trainingmodel.compile(optimizer=sgd, loss=mid_yolo_loss(Ncat = self.categories), metrics=['accuracy'])
        if self.simple == True and self.catsimple == False:
          self.Trainingmodel.compile(optimizer=sgd, loss=simple_yolo_loss(Ncat = self.categories), metrics=['accuracy'])  
        if self.simple == False and self.catsimple == True:
          self.Trainingmodel.compile(optimizer=sgd, loss=cat_simple_yolo_loss(Ncat = self.categories), metrics=['accuracy'])    

        
        self.Trainingmodel.summary()
        print('Training Model:', model_keras)
        
        #Keras callbacks
        lrate = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
        hrate = callbacks.History()
        srate = callbacks.ModelCheckpoint(self.model_dir + self.model_name, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        prate = plotters.PlotHistory(self.Trainingmodel, self.X_val, self.Y_val, self.Categories_Name, plot = self.show, simple = self.simple, catsimple = self.catsimple)
        
        
        #Train the model and save as a h5 file
        self.Trainingmodel.fit(self.X,self.Y, class_weight = d_class_weights , batch_size = self.batch_size, epochs = self.epochs, validation_data=(self.X_val, self.Y_val), shuffle = True, callbacks = [lrate,hrate,srate,prate])

     
        # Removes the old model to be replaced with new model, if old one exists
        if os.path.exists(self.model_dir + self.model_name ):

           os.remove(self.model_dir + self.model_name )
        
        self.Trainingmodel.save(self.model_dir + self.model_name )
        
        
    def plot_prediction(self, idx):
        
        helpers.Printpredict(idx, self.Trainingmodel, self.X_val, self.Y_val, self.Categories_Name)

    
def yolo_loss(Ncat):

    def loss(y_true, y_pred):
        
       
        y_true_class = y_true[...,0:Ncat]
        y_pred_class = y_pred[...,0:Ncat]
        
        
        y_pred_xyt = y_pred[...,Ncat:] 
        
        y_true_xyt = y_true[...,Ncat:] 
        
        
        class_loss = K.mean(K.categorical_crossentropy(y_true_class, y_pred_class), axis=-1)
        xy_loss = K.sum(K.sum(K.square(y_true_xyt - y_pred_xyt), axis = -1), axis = -1)
        
      

        d =  class_loss + xy_loss
        return d 
    return loss
 
def simple_yolo_loss(Ncat):

    def loss(y_true, y_pred):
        
       
        y_true_class = y_true[...,0:Ncat]
        y_pred_class = y_pred[...,0:Ncat]
        
        
        class_loss = K.mean(K.binary_crossentropy(y_true_class, y_pred_class), axis=-1)
      

        d =  class_loss 
        return d 
    return loss       
      
def cat_simple_yolo_loss(Ncat):

    def loss(y_true, y_pred):
         

           y_true_class = y_true[...,0:Ncat]
           y_pred_class = y_pred[...,0:Ncat]

           class_loss = K.mean(K.categorical_crossentropy(y_true_class, y_pred_class), axis = -1)

           d = class_loss
           return d
    return loss   

def mid_yolo_loss(Ncat):
    
    def loss(y_true, y_pred):
        
       
        y_true_class = y_true[...,0:Ncat]
        y_pred_class = y_pred[...,0:Ncat]
        
        
        y_pred_xyt = y_pred[...,Ncat:Ncat + 3] 
        
        y_true_xyt = y_true[...,Ncat:Ncat + 3] 
        
        y_pred_hw = y_pred[...,Ncat + 3:]
        
        y_true_hw = y_true[...,Ncat + 3:]
        
        
        class_loss = K.mean(K.categorical_crossentropy(y_true_class, y_pred_class), axis=-1)
        xy_loss = K.sum(K.sum(K.square(y_true_xyt - y_pred_xyt), axis = -1), axis = -1)
        hw_loss =     K.sum(K.sum(K.square(K.sqrt(y_true_hw) - K.sqrt(y_pred_hw)), axis = -1))
      

        d =  class_loss + 2 * xy_loss + hw_loss
        
        return d 
    return loss
    
        
