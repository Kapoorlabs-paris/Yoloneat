from NEATUtils import plotters
import numpy as np
from NEATUtils import helpers
from NEATUtils.helpers import save_json, load_json, yoloprediction, normalizeFloatZeroOne
from keras import callbacks
import os
import tf
from tqdm import tqdm
from NEATModels import nets, Concat
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
                self.nboxes = config.nboxes
                self.gridx = config.gridx
                self.gridy = config.gridy
                self.gridt = config.gridt
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
                self.lstm_kernel = config['lstm_kernel']
                self.lstm_hidden_unit = config['lstm_hidden_unit']
                self.learning_rate = config['learning_rate']
                self.epochs = config['epochs']
                self.residual = config['residual']
                self.startfilter = config['startfilter']
                self.batch_size = config['batch_size']
                self.multievent = config['multievent']
                self.imagex = config['imagex']
                self.imagey = config['imagey']
                self.imaget = config['size_tminus'] + config['size_tplus'] + 1
                self.nboxes = config['nboxes']
                self.gridx = config['gridx']
                self.gridy = config['gridy']
                self.gridt = config['gridt']
                self.yolo_v0 = config['yolo_v0']
                self.yolo_v1 = config['yolo_v1']
                self.yolo_v2 = config['yolo_v2']
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
        
        self.yololoss = dynamic_yolo_loss(self.categories, self.gridx, self.gridy, self.grid_t, self.nboxes, self.box_vector, self.entropy, self.yolo_v0, self.yolo_v1, self.yolo_v2)
        
        
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
        
        self.Trainingmodel = self.model_keras(input_shape, self.categories,  unit = self.lstm_hidden_unit , box_vector = Y_rest.shape[-1] , depth = self.depth, start_kernel = self.start_kernel, mid_kernel = self.mid_kernel, lstm_kernel = self.lstm_kernel, startfilter = self.startfilter,  input_weights  =  self.model_weights)
        
            
        sgd = optimizers.SGD(lr=self.learning_rate, momentum = 0.99, decay=1e-6, nesterov = True)
        self.Trainingmodel.compile(optimizer=sgd, loss = self.yololoss, metrics=['accuracy'])
        
        self.Trainingmodel.summary()
        
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
        
    def predict(self, imagename, n_tiles = (1,1), overlap_percent = 0.8, event_threshold = 0.5, iou_threshold = 0.01):
        
        self.imagename = imagename
        self.image = imread(imagename)
        self.n_tiles = n_tiles
        self.overlap_percent = overlap_percent
        self.iou_threshold = iou_threshold
        self.event_threshold = event_threshold
        try:
            self.model =  load_model( self.model_dir + self.model_name + '.h5',  custom_objects={'loss':self.yolo_loss, 'Concat':Concat})
        except:
            self.model =  load_model( self.model_dir + self.model_name,  custom_objects={'loss':self.yolo_loss, 'Concat':Concat})
            
        for inputtime in tqdm(0, self.image.shape[0]):
            
            smallimage = self.image[inputtime,:]
            eventboxes = []
            classedboxes = {}
            smallimage = normalizeFloatZeroOne(smallimage,1,99.8)          
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
            
            self.nms()
            self.to_csv()
        
        
    def bbox_iou(self,box1, box2):
        intersect_w = self._interval_overlap([box1['xstart'], box1['xstart'] + box1['xcenter']], [box2['xstart'], box2['xstart'] + box2['xcenter']])
        intersect_h = self._interval_overlap([box1['ystart'], box1['ystart'] + box1['ycenter']], [box2['ystart'], box2['ystart'] + box2['ycenter']])

        intersect = intersect_w * intersect_h

        w1, h1 = box1['width'], box1['height']
        w2, h2 = box2['width'], box2['height']

        union = w1*h1 + w2*h2 - intersect

        return float(np.true_divide(intersect, union))
    
    
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
        for (event_name,event_label) in self.key_categories.items():
            if event_label > 0:
               current_event_box =  self.classedboxes.event_name
               iou_current_event_box = []
               if current_event_box is not None:
                    
                    if len(current_event_box) == 0 :
                        
                        return []
                    else:
                        
                        index_boxes = []  
                        current_event_box = np.array(current_event_box, dtype = float)
                        assert current_event_box.shape[0] > 0
                        if current_event_box.dtype.kind!="f":
                            current_event_box = current_event_box.astype(np.float16)
        
                        idxs = current_event_box[:,event_name].argsort()[::-1]
                        
                        for i in range(len(idxs)):
                            
                            index_i = idxs[i]
                            index_boxes.append(index_i)
                            for j in range(i + 1, len(idxs)):
                                
                                index_j = idxs[j]
                                bbox_iou = self.bbox_iou(current_event_box[index_i], current_event_box[index_j])
                                if bbox_iou >= self.iou_threshold:
                                    
                                    iou_current_event_box.append(current_event_box[index_j])
                                    
               iou_classedboxes[event_name] = [iou_current_event_box]
                                
        self.iou_classedboxes = iou_classedboxes                
        
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
                              
                              iou_current_event_boxes = self.iou_classedboxes.event_name
                              
                              for iou_current_event_box in iou_current_event_boxes:
                                      xcenter = iou_current_event_box['xcenter']
                                      ycenter = iou_current_event_box['ycenter']
                                      tcenter = iou_current_event_box['real_time_event']
                                      confidence = iou_current_event_box['confidence']
                                      angle = iou_current_event_box['angle']
                                      score = iou_current_event_box['event_name']
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
                              writer = csv.writer(open(os.path.dirname(self.imagename) + "/" + event_name + "Location" + (os.path.splitext(os.path.basename(self.imagename))[0])  +".csv", "a"))
                              writer.writerow(['T','Y','X','Score','Size','Confidence','Angle'])
                              for line in event_count:
                                 event_data.append(line)
                                 writer.writerows(event_data)
                                 event_data = []           
                              
            
          
    def overlaptiles(self):
        
            if self.n_tiles == 1:
                
                       patchshape = (self.image.shape[1], self.image.shape[2])  
                      
                       image = zero_pad(self.image, self.stride,self.stride)
        
                       patch = []
                       rowout = []
                       column = []
                       
                       patch.append(image)
                       rowout.append(0)
                       column.append(0)
                     
            else:
                  
             patchx = self.image.shape[2] // self.n_tiles
             patchy = self.image.shape[1] // self.n_tiles
        
             if patchx > self.imagex and patchy > self.imagey:
              if self.overlap_percent > 1 or self.overlap_percent < 0:
                 self.overlap_percent = 0.8
             
              jumpx = int(self.overlap_percent * patchx)
              jumpy = int(self.overlap_percent * patchy)
             
              patchshape = (patchy, patchx)   
              rowstart = 0; colstart = 0
              pairs = []  
              #row is y, col is x
              
              while rowstart < self.image.shape[1] - patchy:
                 colstart = 0
                 while colstart < self.image.shape[2] - patchx:
                    
                     # Start iterating over the tile with jumps = stride of the fully convolutional network.
                     pairs.append([rowstart, colstart])
                     colstart+=jumpx
                 rowstart+=jumpy 
                
              #Include the last patch   
              rowstart = self.image.shape[1] - patchy
              colstart = 0
              while colstart < self.image.shape[2] - patchx:
                            pairs.append([rowstart, colstart])
                            colstart+=jumpx
              rowstart = 0
              colstart = self.image.shape[2] - patchx
              while rowstart < self.image.shape[1] - patchy:
                            pairs.append([rowstart, colstart])
                            rowstart+=jumpy              
                            
              if self.image.shape[1] >= self.imagey and self.image.shape[2]>= self.imagex :          
                  
                    patch = []
                    rowout = []
                    column = []
                    for pair in pairs: 
                       smallpatch, smallrowout, smallcolumn =  chunk_list(self.image, patchshape, self.stride, pair)
                       patch.append(smallpatch)
                       rowout.append(smallrowout)
                       column.append(smallcolumn) 
                
             else:
                 
                       patch = []
                       rowout = []
                       column = []
                       image = zero_pad(self.image, self.stride,self.stride)
                       
                       patch.append(image)
                       rowout.append(0)
                       column.append(0)
                       
            self.patch = patch          
            self.sy = rowout
            self.sx = column            
          
        
    def predict_main(self,sliceregion):
            try:
                self.overlaptiles()
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


            region = (slice(rowstart, endrow),
                      slice(colstart, endcol))

            # The actual pixels in that region.
            patch = image[region]

            # Always normalize patch that goes into the netowrk for getting a prediction score 
            patch = zero_pad(patch, stride, stride)


            return patch, rowstart, colstart       
   