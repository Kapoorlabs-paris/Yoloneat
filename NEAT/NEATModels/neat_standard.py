from NEATUtils import plotters
import numpy as np
from NEATUtils import helpers
from NEATUtils.helpers import save_json, load_json, yoloprediction, normalizeFloatZeroOne
from keras import callbacks
import os
import tensorflow as tf
from tqdm import tqdm
from NEATModels import nets
from NEATModels.nets import Concat
from NEATModels.loss import dynamic_yolo_loss
from keras import backend as K
#from IPython.display import clear_output
from keras import optimizers
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path
from keras.models import load_model
from tifffile import imread, imwrite
import csv


class NEATDynamic(object):
    

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

        
        self.config = config
        self.catconfig = catconfig
        self.cordconfig = cordconfig
        self.model_dir = model_dir
        self.model_name = model_name 
        if self.config !=None:
                self.npz_directory = config.npz_directory
                self.npz_name = config.npz_name
                self.npz_val_name = config.npz_val_name
                self.key_categories = config.key_categories
                self.box_vector = config.box_vector
                self.show = config.show
                self.key_cord = config.key_cord
                self.categories = len(config.key_categories)
                self.depth = config.depth
                self.start_kernel = config.start_kernel
                self.mid_kernel = config.mid_kernel
                self.lstm_kernel = config.lstm_kernel
                self.learning_rate = config.learning_rate
                self.epochs = config.epochs
                self.residual = config.residual
                self.startfilter = config.startfilter
                self.batch_size = config.batch_size
                self.multievent = config.multievent
                self.imagex = config.imagex
                self.imagey = config.imagey
                self.imaget = config.size_tminus + config.size_tplus + 1
                self.size_tminus = config.size_tminus
                self.size_tplus = config.size_tplus
                self.nboxes = config.nboxes
                self.gridx = 1
                self.gridy = 1
                self.gridt = 1
                self.yolo_v0 = config.yolo_v0
                self.yolo_v1 = config.yolo_v1
                self.yolo_v2 = config.yolo_v2
                self.stride = config.stride
                self.lstm_hidden_unit = config.lstm_hidden_unit
        if self.config == None:
               
                try:
                   self.config = load_json(self.model_dir + os.path.splitext(self.model_name)[0] + '_Parameter.json')
                except:
                   self.config = load_json(self.model_dir + self.model_name + '_Parameter.json')  
                   
                self.npz_directory = self.config['npz_directory']
                self.npz_name = self.config['npz_name']
                self.npz_val_name = self.config['npz_val_name']
                self.key_categories = self.catconfig
                self.box_vector = self.config['box_vector']
                self.show = self.config['show']
                self.key_cord = self.cordconfig
                self.categories = len(self.catconfig)
                self.depth = self.config['depth']
                self.start_kernel = self.config['start_kernel']
                self.mid_kernel = self.config['mid_kernel']
                self.lstm_kernel = self.config['lstm_kernel']
                self.lstm_hidden_unit = self.config['lstm_hidden_unit']
                self.learning_rate = self.config['learning_rate']
                self.epochs = self.config['epochs']
                self.residual = self.config['residual']
                self.startfilter = self.config['startfilter']
                self.batch_size = self.config['batch_size']
                self.multievent = self.config['multievent']
                self.imagex = self.config['imagex']
                self.imagey = self.config['imagey']
                self.imaget = self.config['size_tminus'] + self.config['size_tplus'] + 1
                self.size_tminus = self.config['size_tminus']
                self.size_tplus = self.config['size_tplus']
                self.nboxes = self.config['nboxes']
                self.gridx = 1
                self.gridy = 1
                self.gridt = 1
                self.yolo_v0 = self. config['yolo_v0']
                self.yolo_v1 = self.config['yolo_v1']
                self.yolo_v2 = self.config['yolo_v2']
                self.stride = self.config['stride']   
                self.lstm_hidden_unit = self.config['lstm_hidden_unit']
                
                
 
        self.X = None
        self.Y = None
        self.axes = None
        self.X_val = None
        self.Y_val = None
        self.Trainingmodel = None
        self.Xoriginal = None
        self.Xoriginal_val = None
        
        if self.residual:
            self.model_keras = nets.ORNET
        else:
            self.model_keras = nets.OSNET
            
        if self.multievent == True:
           self.last_activation = 'sigmoid'
           self.entropy = 'binary'
           
           
        if self.multievent == False:
           self.last_activation = 'softmax'              
           self.entropy = 'notbinary' 
        
        self.yololoss = dynamic_yolo_loss(self.categories, self.gridx, self.gridy, self.gridt, self.nboxes, self.box_vector, self.entropy, self.yolo_v0, self.yolo_v1, self.yolo_v2)
        
        
    def loadData(self):
        
        (X,Y),  axes = helpers.load_full_training_data(self.npz_directory, self.npz_name, verbose= True)

        (X_val,Y_val), axes = helpers.load_full_training_data(self.npz_directory, self.npz_val_name,  verbose= True)
        
        
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
        
        model_weights = self.model_dir + self.model_name
        if os.path.exists(model_weights):
        
            self.model_weights = model_weights
            print('loading weights')
        else:
           
            self.model_weights = None
        
        print(self.box_vector)
        dummyY = np.zeros([self.Y.shape[0],self.Y.shape[1],self.Y.shape[2],self.categories + self.nboxes* self.box_vector])
        dummyY[:,:,:,:self.Y.shape[3]] = self.Y
        
        dummyY_val = np.zeros([self.Y_val.shape[0],self.Y_val.shape[1],self.Y_val.shape[2],self.categories + self.nboxes* self.box_vector])
        dummyY_val[:,:,:,:self.Y_val.shape[3]] = self.Y_val
        for b in range(1, self.nboxes):
            
            dummyY[:,:,:,self.categories + b * self.box_vector:self.categories + (b + 1) * self.box_vector] = self.Y[:,:,:, self.categories: self.categories + self.box_vector]
            dummyY_val[:,:,:,self.categories + b * self.box_vector:self.categories + (b + 1) * self.box_vector] = self.Y_val[:,:,:, self.categories: self.categories + self.box_vector]
            
        self.Y = dummyY
        self.Y_val = dummyY_val
        
        print(self.Y.shape, self.nboxes)
        
        
        self.Trainingmodel = self.model_keras(input_shape, self.categories,  unit = self.lstm_hidden_unit , box_vector = Y_rest.shape[-1] , depth = self.depth, start_kernel = self.start_kernel, mid_kernel = self.mid_kernel, lstm_kernel = self.lstm_kernel, startfilter = self.startfilter,  input_weights  =  self.model_weights)
        
            
        sgd = optimizers.SGD(lr=self.learning_rate, momentum = 0.99, decay=1e-6, nesterov = True)
        self.Trainingmodel.compile(optimizer=sgd, loss = self.yololoss, metrics=['accuracy'])
        
        self.Trainingmodel.summary()
        
        #Keras callbacks
        lrate = callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=4, verbose=1)
        hrate = callbacks.History()
        srate = callbacks.ModelCheckpoint(self.model_dir + self.model_name, monitor='loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)
        prate = plotters.PlotHistory(self.Trainingmodel, self.X_val, self.Y_val, self.key_categories, self.key_cord, self.gridx, self.gridy, plot = self.show, nboxes = self.nboxes)
        
        
        #Train the model and save as a h5 file
        self.Trainingmodel.fit(self.X,self.Y, class_weight = d_class_weights , batch_size = self.batch_size, epochs = self.epochs, validation_data=(self.X_val, self.Y_val), shuffle = True, callbacks = [lrate,hrate,srate,prate])

     
        # Removes the old model to be replaced with new model, if old one exists
        if os.path.exists(self.model_dir + self.model_name ):

           os.remove(self.model_dir + self.model_name )
        
        self.Trainingmodel.save(self.model_dir + self.model_name )
        
    def predict(self, imagename, n_tiles = (1,1), overlap_percent = 0.8, event_threshold = 0.5, iou_threshold = 0.01):
        
        self.imagename = imagename
        self.image = imread(imagename)
        self.n_tiles = n_tiles
        self.overlap_percent = overlap_percent
        self.iou_threshold = iou_threshold
        self.event_threshold = event_threshold
        try:
            self.model =  load_model( self.model_dir + self.model_name + '.h5',  custom_objects={'loss':self.yololoss, 'Concat':Concat})
        except:
            self.model =  load_model( self.model_dir + self.model_name,  custom_objects={'loss':self.yololoss, 'Concat':Concat})
            
        eventboxes = []
        classedboxes = {}    
        count = 0
        for inputtime in tqdm(range(0, self.image.shape[0])):
            if inputtime < self.image.shape[0] - self.imaget:
                       
                        count = count + 1
                        smallimage = CreateVolume(self.image, self.imaget, inputtime,self.imagex, self.imagey)
                        
                        #Break image into tiles if neccessary
                        predictions, allx, ally = self.predict_main(smallimage)
                        #Iterate over tiles
                        for p in range(0,len(predictions)):   
                
                          sum_time_prediction = predictions[p]
                          
                          if sum_time_prediction is not None:
                             #For each tile the prediction vector has shape N H W Categories + Trainng Vector labels
                             for i in range(0, sum_time_prediction.shape[0]):
                                  time_prediction =  sum_time_prediction[i]
                                  boxprediction = yoloprediction(smallimage, ally[p], allx[p], time_prediction, self.stride, inputtime, self.config, self.key_categories, self.key_cord, self.nboxes, 'detection', 'dynamic')
                                  
                                  if boxprediction is not None:
                                          eventboxes = eventboxes + boxprediction
                                     
                        for (event_name,event_label) in self.key_categories.items(): 
                                 
                            if event_label > 0:
                                 current_event_box = []
                                 for box in eventboxes:
                            
                                    event_prob = box[event_name]
                                    if event_prob > self.event_threshold:
                                       
                                        current_event_box.append(box)
                                 classedboxes[event_name] = [current_event_box]
                             
                        self.classedboxes = classedboxes    
                        self.eventboxes =  eventboxes
                        
                        if count == self.imaget:
                            #nms over time
                            self.nms()
                            self.to_csv()
                            eventboxes = []
                            classedboxes = {}    
                            count = 0
                
        
    def bbox_iou(self,box1, box2):
        
        
        w1, h1 = box1['width'], box1['height']
        w2, h2 = box2['width'], box2['height']
        
        xA =max( box1['xstart'] , box2['xstart'] )
        xB = max ( box1['xstart'] + w1, box2['xstart'] + w2)
        yA = max( box1['ystart'] , box2['ystart'] )
        yB = max (box1['ystart'] + h1, box2['ystart'] + h2)

        intersect = max(0, xB - xA) * max(0, yB - yA)



        union = w1*h1 + w2*h2 - intersect

        return float(np.true_divide(intersect, union)) - 1
    
    
    def _interval_overlap(self,interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                 return 0
            else:
                return min(x2,x4) - x3
        
    def nms(self):
        
        
        iou_classedboxes = {}
        best_iou_classedboxes = {}
        self.iou_classedboxes = {}
        for (event_name,event_label) in self.key_categories.items():
            if event_label > 0:
               #Get all events
               event_box = self.classedboxes[event_name][0]
               
               #Highest probability is first
               sorted_event_box = sorted(event_box, key = lambda k : k[event_name], reverse = True)
               iou_current_event_box = []
               best_iou_current_event_box = []
               poppedj = []
               
               
               if sorted_event_box is not None:
                    
                    if len(sorted_event_box) == 0 :
                        
                        return []
                    else:
                        
                        #fIRST ROUND
                        for i in range(len(sorted_event_box)):
                            
                            best_iou = []
                            for j in range(i + 1, len(sorted_event_box)):
                                if j not in poppedj:
                                        bbox_iou = self.bbox_iou(sorted_event_box[i], sorted_event_box[j])
                                        if bbox_iou >= 0.1:
                                             best_iou.append(bbox_iou)
                                        if bbox_iou < self.iou_threshold:
                                            
                                              poppedj.append(j)
                                        #good event found     
                                        if len(best_iou) > 10:
                                            if sorted_event_box[i] not in iou_current_event_box:
                                                iou_current_event_box.append(sorted_event_box[i])
                                            
                                    
                                    
                                    
               iou_classedboxes[event_name] = [iou_current_event_box]
               
               #lAST ROUND
               for i in range(len(iou_current_event_box)):
                            best_iou = []
                            for j in range(i + 1, len(iou_current_event_box)):
                                
                                        bbox_iou = self.bbox_iou(iou_current_event_box[i], iou_current_event_box[j])
                                       
                                        if bbox_iou > self.iou_threshold:
                                            
                                            #EXTRA good event found     
                                            if iou_current_event_box[i] not in best_iou_current_event_box:
                                                best_iou_current_event_box.append(iou_current_event_box[i])
               
               best_iou_classedboxes[event_name] = [best_iou_current_event_box]                
        self.iou_classedboxes = best_iou_classedboxes                
        
    def to_csv(self):
        
        for (event_name,event_label) in self.key_categories.items():
                   
                   if event_label > 0:
                              xlocations = []
                              ylocations = []
                              scores = []
                              confidences = []
                              tlocations = []   
                              radiuses = []
                              angles = []
                              try:
                                      iou_current_event_boxes = self.iou_classedboxes[event_name][0]
                                      iou_current_event_boxes = sorted(iou_current_event_boxes, key = lambda x:x[event_name], reverse = True) 
                                      for iou_current_event_box in iou_current_event_boxes:
                                              xcenter = iou_current_event_box['xcenter']
                                              ycenter = iou_current_event_box['ycenter']
                                              tcenter = iou_current_event_box['real_time_event']
                                              confidence = iou_current_event_box['confidence']
                                              angle = iou_current_event_box['realangle']
                                              score = iou_current_event_box[event_name]
                                              radius = np.sqrt( iou_current_event_box['height'] * iou_current_event_box['height'] + iou_current_event_box['width'] * iou_current_event_box['width']  )// 2
                                              xlocations.append(xcenter)
                                              ylocations.append(ycenter)
                                              scores.append(score)
                                              confidences.append(confidence)
                                              tlocations.append(tcenter)
                                              radiuses.append(radius)
                                              angles.append(angle)
                                    
                                      
                                      event_count = np.column_stack([tlocations,ylocations,xlocations,scores,radiuses,confidences,angles]) 
                                      event_data = []
                                      csvname = os.path.dirname(self.imagename) + "/" + event_name + "Location" + (os.path.splitext(os.path.basename(self.imagename))[0])
                                      writer = csv.writer(open(csvname  +".csv", "a"))
                                      filesize = os.stat(csvname + ".csv").st_size
                                      if filesize < 1:
                                         writer.writerow(['T','Y','X','Score','Size','Confidence','Angle'])
                                      for line in event_count:
                                         if line not in event_data:  
                                            event_data.append(line)
                                         writer.writerows(event_data)
                                         event_data = []           
                              
                              except: 
                                  pass
          
    def overlaptiles(self, sliceregion):
        
            if self.n_tiles == 1:
                
                       patchshape = (sliceregion.shape[1], sliceregion.shape[2])  
                      
                       image = zero_pad(sliceregion, self.stride,self.stride)
        
                       patch = []
                       rowout = []
                       column = []
                       
                       patch.append(image)
                       rowout.append(0)
                       column.append(0)
                     
            else:
                  
             patchx = sliceregion.shape[2] // self.n_tiles[0]
             patchy = sliceregion.shape[1] // self.n_tiles[1]
        
             if patchx > self.imagex and patchy > self.imagey:
              if self.overlap_percent > 1 or self.overlap_percent < 0:
                 self.overlap_percent = 0.8
             
              jumpx = int(self.overlap_percent * patchx)
              jumpy = int(self.overlap_percent * patchy)
             
              patchshape = (patchy, patchx)   
              rowstart = 0; colstart = 0
              pairs = []  
              #row is y, col is x
              
              while rowstart < sliceregion.shape[1] - patchy:
                 colstart = 0
                 while colstart < sliceregion.shape[2] - patchx:
                    
                     # Start iterating over the tile with jumps = stride of the fully convolutional network.
                     pairs.append([rowstart, colstart])
                     colstart+=jumpx
                 rowstart+=jumpy 
                
              #Include the last patch   
              rowstart = sliceregion.shape[1] - patchy
              colstart = 0
              while colstart < sliceregion.shape[2] - patchx:
                            pairs.append([rowstart, colstart])
                            colstart+=jumpx
              rowstart = 0
              colstart = sliceregion.shape[2] - patchx
              while rowstart < sliceregion.shape[1] - patchy:
                            pairs.append([rowstart, colstart])
                            rowstart+=jumpy              
                            
              if sliceregion.shape[1] >= self.imagey and sliceregion.shape[2]>= self.imagex :          
                  
                    patch = []
                    rowout = []
                    column = []
                    for pair in pairs: 
                       smallpatch, smallrowout, smallcolumn =  chunk_list(sliceregion, patchshape, self.stride, pair)
                       patch.append(smallpatch)
                       rowout.append(smallrowout)
                       column.append(smallcolumn) 
                
             else:
                 
                       patch = []
                       rowout = []
                       column = []
                       image = zero_pad(sliceregion, self.stride,self.stride)
                       
                       patch.append(image)
                       rowout.append(0)
                       column.append(0)
            self.patch = patch          
            self.sy = rowout
            self.sx = column            
          
        
    def predict_main(self,sliceregion):
            try:
                self.overlaptiles(sliceregion)
                predictions = []
                allx = []
                ally = []
                for i in range(0,len(self.patch)):   
                   
                   sum_time_prediction = self.make_patches(self.patch[i])

                   predictions.append(sum_time_prediction)
                   allx.append(self.sx[i])
                   ally.append(self.sy[i])
           
            except tf.errors.ResourceExhaustedError:
                
                print('Out of memory, increasing overlapping tiles for prediction')
                
                self.n_tiles = self.n_tiles  + 1
                
                print('Tiles: ', self.n_tiles)
                
                self.predict_main(sliceregion)
                
            return predictions, allx, ally
        
    def make_patches(self, sliceregion):
       
       
       predict_im = np.expand_dims(sliceregion,0)
       
       
       prediction_vector = self.model.predict(np.expand_dims(predict_im,-1), verbose = 0)
         
            
       return prediction_vector
   
    
def zero_pad(patch, jumpx, jumpy):

          time = patch.shape[0]
          sizeY = patch.shape[1]
          sizeX = patch.shape[2]
          sizeXextend = sizeX
          sizeYextend = sizeY
         
 
          while sizeXextend%jumpx!=0:
              sizeXextend = sizeXextend + 1
        
          while sizeYextend%jumpy!=0:
              sizeYextend = sizeYextend + 1

          extendimage = np.zeros([time, sizeYextend, sizeXextend])
          
          extendimage[0:time, 0:sizeY, 0:sizeX] = patch
              
          return extendimage
      
        
def chunk_list(image, patchshape, stride, pair):
            rowstart = pair[0]
            colstart = pair[1]

            endrow = rowstart + patchshape[0]
            endcol = colstart + patchshape[1]

            if endrow > image.shape[1]:
                endrow = image.shape[1]
            if endcol > image.shape[2]:
                endcol = image.shape[2]


            region = (slice(0,image.shape[0]),slice(rowstart, endrow),
                      slice(colstart, endcol))

            # The actual pixels in that region.
            patch = image[region]

            # Always normalize patch that goes into the netowrk for getting a prediction score 
            patch = normalizeFloatZeroOne(patch,1,99.8)
            patch = zero_pad(patch, stride, stride)


            return patch, rowstart, colstart
        
        
def CreateVolume(patch, imaget, timepoint, imagey, imagex):
    
               starttime = timepoint
               endtime = timepoint + imaget
               smallimg = patch[starttime:endtime, :]
       
               return smallimg         
       